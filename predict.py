#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:32:16 2025

@author: na19
"""

from imports import *

def loadModel(path, model_instance, fileName, total_path):
    """
    Loads trained model for prediction.

    Parameters:
        path (str): Path to saved trained models.
        model_instance: Initialized model instance (e.g., Unet, DnCNN).
        fileName (str): Name of the saved model file.
        total_path (str): Full model path (usually path + filename).

    Returns:
        epoch (int): Epoch number where model was saved.
        model: Loaded model with weights.
        optimiser: Optimizer with loaded state.
    """
    total_path = os.path.join(path, fileName)

    # Load model weights and optimizer state
    model = model_instance.to(device)
    checkpoint = torch.load(total_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    optimiser = optim.Adam(model.parameters(), lr=1e-5)
    optimiser.load_state_dict(checkpoint['optimizer'])

    epoch = checkpoint['epoch']
    return epoch, model, optimiser


def predict_mat_each_slice_perf(patient_path, pat_name, img_name, mag_path_test):
    """
    Predict magnitude images slice-by-slice using the trained model.

    Parameters:
        patient_path (str): Test patient directory containing .mat files.
        pat_name (str): Patient's name.
        img_name (str): Image filename.
        mag_path_test (str): Path to the magnitude dataset (.mat file).

    Returns:
        mag_path (str): Path where denoised magnitude predictions are saved.
    """
    # Load magnitude image from .mat file
    mat_img = loadmat(mag_path_test)
    mag_img_test = mat_img['magImages']

    new_arr = torch.from_numpy(mag_img_test.astype(float)).float().to(device)
    pred_arr = np.zeros_like(mag_img_test)
    dn = np.zeros_like(mag_img_test)

    x_mag = []

    # Iterate over slices (4th dimension)
    for i in range(new_arr.shape[3]):
        # Extract single slice and add batch and channel dims
        og_arr = torch.from_numpy(mag_img_test[:, :, 0, i].astype(float)).float().unsqueeze(0).unsqueeze(0).to(device)

        # Standardize image slice
        std, mean = torch.std_mean(og_arr, dim=(-2, -1), unbiased=False)
        std = std.reshape(std.shape + (1, 1))
        new_arr_norm = (og_arr - mean) / std

        # Predict residual noise with the model
        residual = model(new_arr_norm)
        pred_arr = residual * std

        # Calculate denoised image slice: original - predicted residual
        dn_slice = (og_arr.cpu().detach().numpy()) - (pred_arr.cpu().detach().numpy())
        x_mag.append(dn_slice)

        # Uncomment for visualization if needed:
        # plt.figure(i, figsize=(15, 15))
        # plt.subplot(121)
        # plt.imshow(x_mag_og[i], cmap='gray')
        # plt.subplot(122)
        # plt.imshow(np.abs(complex_arr), cmap='gray')
        # plt.subplot(143)
        # plt.imshow(x_phase_og[i], cmap='gray')
        # plt.subplot(144)
        # plt.imshow(np.mod(np.angle(complex_arr), 2 * np.pi), cmap='gray')

    # Convert list to numpy array and reorder axes as required
    x_mag = np.array(x_mag)
    x_mag = torch.from_numpy(x_mag).squeeze(1).permute(2, 3, 1, 0).detach().numpy()

    # Define output directory and filename
    mag_path = os.path.join(patient_path, pat_name, 'MAT_GRAPPA_DENOISED')
    os.makedirs(mag_path, exist_ok=True)

    filename = os.path.basename(os.path.normpath(os.path.join(patient_path, pat_name, 'MAT_GRAPPA_MOCO', img_name)))

    # Save the denoised magnitude images to .mat file
    savemat(
        os.path.join(mag_path, filename),
        {
            '__header__': mat_img['__header__'],
            '__version__': mat_img['__version__'],
            '__globals__': mat_img['__globals__'],
            'magImages': x_mag
        }
    )

    torch.cuda.empty_cache()
    return mag_path


# Initialize pretrained model
model = BasicUNet(
    spatial_dims=2, in_channels=1, out_channels=1,
    features=(64, 128, 256, 512, 1024, 128),
    act=('ReLU', {'inplace': True}),
    norm=('batch', {'affine': True}),
    bias=False,
    dropout=0.5,
    upsample='nontrainable'
)
model = model.to(device)

# Paths for model loading
base_dir = '/media/na19/DATA/Naledi/denoising_data/Trained_models/Frontiers_noisemapnet/original_data_models/best_model/'
model_name = "noisemapnetogNL2513model_trial_0.pth"
total_path = os.path.join(base_dir, model_name)

# Load trained model checkpoint
epoch, model, optimiser = loadModel(base_dir, model, model_name, total_path)
model.eval()

# Prediction loop over patients and image slices
patient_path = '/media/na19/DATA/Naledi/denoising_data/Data/cmr_trainset/CINE DATA/Frontiers_datasets_080124/Perfusion_test_examples/Frontiers_analysis/Noise_level_predictions/NL25/'

for patient in os.listdir(patient_path):
    print('Processing patient:', patient)
    mag_dir = os.path.join(patient_path, patient, 'MAT_GRAPPA_MOCO')

    if not os.path.isdir(mag_dir):
        continue  # Skip if directory doesn't exist

    for slice_name in os.listdir(mag_dir):
        mag_test_img_path = os.path.join(mag_dir, slice_name)
        print(f"Predicting: {mag_test_img_path}")

        # Call prediction function
        _ = predict_mat_each_slice_perf(patient_path, patient, slice_name, mag_test_img_path)

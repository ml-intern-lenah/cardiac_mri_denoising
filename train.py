#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 00:34:29 2025

@author: na19
"""

from imports import *
from models import DnCNN
from monai.networks.nets import BasicUNet


def train_model(trial, model, criterion, optimiser, train_loader, val_loader, best_model_path, epochs=3):
    """
    Training loop for models

    Parameters:
        trial (obejct): objective function
        model (str): denoising model being trained
        criterion (str): loss function used
        optimiser (str): optimiser name- Adam/SGD
        train_loadder: loaded training data
        val_loader: loaded validation data
        best_model_path(str): path to save best model
        epochs: number of epochs
       
    Returns (floats): training and validation loses from each trained model
    """

    print('Training model:', model.__class__.__name__)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    training_losses = []
    validation_losses = []
    epochs_arr = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for ground_truth_label in train_loader:
            ground_truth_label = ground_truth_label.permute(1, 0, 2, 3).to(device)

            gauss_noise = torch.FloatTensor(ground_truth_label.size()).normal_(mean=0, std=12.5).to(device)
            noisy_image = ground_truth_label + gauss_noise

            std, mean = torch.std_mean(noisy_image, dim=(-2, -1), unbiased=False)
            std = std.reshape(std.shape + (1, 1)).to(device)
            mean = mean.reshape(mean.shape + (1, 1)).to(device)

            noisy_image_norm = (noisy_image - mean) / std
            gauss_noise_norm = gauss_noise / std

            noisy_image_norm = noisy_image_norm.float()
            ground_truth_label = ground_truth_label.float()
            gauss_noise_norm = gauss_noise_norm.float()

            optimiser.zero_grad()
            predicted_noise_label = model(noisy_image_norm)
            loss = criterion(predicted_noise_label, gauss_noise_norm)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * ground_truth_label.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_train_loss)
        

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for ground_truth_val in val_loader:
                ground_truth_val = ground_truth_val.permute(1, 0, 2, 3).to(device)
                gauss_val_noise = torch.FloatTensor(ground_truth_val.size()).normal_(mean=0, std=12.5).to(device)
                noisy_val_img = ground_truth_val + gauss_val_noise

                stdv, meanv = torch.std_mean(noisy_val_img, dim=(-2, -1), unbiased=False)
                stdv = stdv.reshape(stdv.shape + (1, 1)).to(device)
                meanv = meanv.reshape(meanv.shape + (1, 1)).to(device)

                noisy_val_img_norm = (noisy_val_img - meanv) / stdv
                gauss_val_noise_norm = gauss_val_noise / stdv

                predicted_noisy_img = model(noisy_val_img_norm.float())
                val_loss = criterion(predicted_noisy_img, gauss_val_noise_norm)

                val_running_loss += val_loss.item() * ground_truth_val.size(0)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model_filename = f"{best_model_path}/best_model_epoch_{epoch}.pth"
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimiser.state_dict()}, model_filename)
                    print(f"Saved new best model at epoch {epoch} with val loss {val_loss.item():.4f}")

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        validation_losses.append(epoch_val_loss)

        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    # save epochs and loses as arrays
    os.makedirs(best_model_path, exist_ok=True)
    epochs_arr = np.arange(1, epochs+1)
    # Save losses as text files
    np.savetxt(os.path.join(best_model_path, 'training_losses.txt'), np.array(training_losses))
    np.savetxt(os.path.join(best_model_path, 'validation_losses.txt'), np.array(validation_losses))
    np.savetxt(os.path.join(best_model_path, 'epochs.txt'), epochs_arr)
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_val_loss:.6f}')

    model.load_state_dict(best_model_wts)

    return training_losses, validation_losses, best_val_loss


 # all lossses dictionary
all_losses= {}
def objective(trial, train_loader, val_loader, hyperparams):
    """
    Uses optuna to find best parameters and best performing model

    Parameters:
        trial (object): objective function
        train_loader: loaded trainign dataset
        val_loader: loaded validation dataset
        hyperparams: set model hyper parameters
        
    Returns (floats): best Unet and DnCNN
    """
    
   
    # Suggest hyperparameters as per your sequence
    params = {
        "lr": trial.suggest_loguniform('lr', 1e-5, 1e-2),
        "batchsize": trial.suggest_int('batchsize', 1, 16),
        "noiselevel": trial.suggest_float("noiselevel", 6.5, 25),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "optimizer_name": trial.suggest_categorical('optimizer_name', ["Adam", "SGD"]),
        "model_name": trial.suggest_categorical("model_name", ["UNet", "DnCNN"])
    }
    global all_losses

    
    # Criterion and optimizer class
    criterion = nn.MSELoss()
    optimizer_class = getattr(torch.optim, params["optimizer_name"])

    # Initialize UNet
    unet = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=(64, 128, 256, 512, 1024, 128),
        act=('ReLU', {'inplace': True}),
        norm=('batch', {'affine': True}),
        bias=False,
        dropout=params["dropout"],
        upsample='nontrainable'
    ).to(device)

    # Initialize DnCNN
    dncnn = DnCNN().to(device)

    # Optimizers
    optimizer_unet = optimizer_class(unet.parameters(), lr=params["lr"])
    optimizer_dncnn = optimizer_class(dncnn.parameters(), lr=params["lr"])

    # Model saving paths
    base_path = "/media/na19/DATA/Naledi/denoising_data/Trained_models/DnCNNvs_dnet/"
    path_unet = os.path.join(base_path, "unet/")
    path_dncnn = os.path.join(base_path, "dncnn/")
    os.makedirs(path_unet, exist_ok=True)
    os.makedirs(path_dncnn, exist_ok=True)

    # Train UNet
    train_losses_unet, val_losses_unet, best_val_unet = train_model(
        trial=trial,
        model=unet,
        criterion=criterion,
        optimiser=optimizer_unet,
        train_loader=train_loader,
        val_loader=val_loader,
        best_model_path=path_unet,
        epochs=hyperparams["num_epochs"]
    )

    # Train DnCNN
    train_losses_dncnn, val_losses_dncnn, best_val_dncnn = train_model(
        trial=trial,
        model=dncnn,
        criterion=criterion,
        optimiser=optimizer_dncnn,
        train_loader=train_loader,
        val_loader=val_loader,
        best_model_path=path_dncnn,
        epochs=hyperparams["num_epochs"]
    )

    # Save losses globally for plotting later
    all_losses.clear()
    all_losses.update({
        "unet": {"train": train_losses_unet, "val": val_losses_unet},
        "dncnn": {"train": train_losses_dncnn, "val": val_losses_dncnn}
    })

    # Return best validation loss between both models
    return min(best_val_unet, best_val_dncnn)
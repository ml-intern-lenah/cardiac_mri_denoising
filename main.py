#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 23:18:39 2025

@author: na19
"""

from imports import *
from data_processing import NumpyDataset, NumpyDataset_non_mri

# models
from models import DnCNN
from monai.networks.nets import BasicUNet

# training and Optuna optimisation
from train import * #train_model, objective
from predict import *


def main():
    
    best_model_path = "/media/na19/DATA/Naledi/denoising_data/Trained_models/Frontiers_noisemapnet/original_data_models/best_model/"
    hyperparams  = hyper_params() # btained from imports
    
    #%% -------------------------------------dataloading-------------------------------#

    # data directories
    dataset_train= NumpyDataset(clean_image_folder="/media/na19/DATA/Naledi/denoising_data/Data/cmr_trainset/CINE DATA/Frontiers_datasets_080124/cine_patients/renamed_trainset/",  hyperparams=hyperparams, transform =None)
    datasetval= NumpyDataset(clean_image_folder="/media/na19/DATA/Naledi/denoising_data/Data/cmr_trainset/CINE DATA/Frontiers_datasets_080124/cine_patients/renamed_valset/",  hyperparams=hyperparams, transform =None)

    # 3T datasest directory
    #dataset_train= NumpyDataset(clean_image_folder="/media/na19/DATA/Naledi/denoising_data/Data/cmr_trainset/CINE DATA/Frontiers_datasets_080124/SHAX_Fontiers/3T_shax_train_renamed/",transform =None)
    #datasetval= NumpyDataset(clean_image_folder="/media/na19/DATA/Naledi/denoising_data/Data/cmr_trainset/CINE DATA/Frontiers_datasets_080124/SHAX_Fontiers/3T_shax_val_renamed/", transform =None)

    ## NON-MRI datasest-random dataserr
    #dataset_train= NumpyDataset_non_mri(clean_image_folder="/media/na19/DATA/Naledi/denoising_data/Data/cmr_trainset/CINE DATA/Frontiers_datasets_080124/nonmri/train/",transform =None)
    #datasetval= NumpyDataset_non_mri(clean_image_folder="/media/na19/DATA/Naledi/denoising_data/Data/cmr_trainset/CINE DATA/Frontiers_datasets_080124/nonmri/validation/", transform =None)


     # Create a dataloader instance- takes clean image and it's corresponding noisy image label
    batch_size=1
    print(len(dataset_train))
    print(len(datasetval))
    indices = list(range(len(dataset_train)))

    # Train dataset indices
    train_size = len(dataset_train)
    train_indices = list(range(train_size))
    val_size = len(datasetval)
    val_indices = list(range(val_size))

    # Shuffle and split within each dataset separately (if you want to split further)

    # Example: no split, just create samplers directly:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset_train, sampler=train_sampler, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(datasetval, sampler=valid_sampler, batch_size=batch_size, num_workers=8, pin_memory=True)

    #%%  -------------------------------------tDenoising UNet/ NOiseMapNet-------------------------------#

    # define class U-Net model
    dnet = BasicUNet(spatial_dims=2, in_channels=1, out_channels=1,features=(64, 128, 256, 512, 1024, 128),act=('ReLU', {'inplace': True}),norm=('batch', {'affine': True}),bias=False,dropout=0.5, upsample='nontrainable')
    dnet = dnet.to(device)
    print(dnet)

    #%%  -------------------------------------DnCNN Comparison- -----------------------------#
    dncnn_model = DnCNN().to(device)

    #------------------------------------training-------------------------------#
    # loss function
    criterion = nn.MSELoss()

    # Defining the optimiser and it's parameters
    optimiser_dnet= torch.optim.Adam(filter(lambda p: p.requires_grad, dnet.parameters()), lr=1e-1, betas=(0.9, 0.999)) # adjust UNet for DnCNNN 28/06/25
    # adding DnCNN 28/06/25
    optimiser_DnCNN = torch.optim.Adam(filter(lambda p: p.requires_grad, dncnn_model.parameters()), lr=1e-1, betas=(0.9, 0.999)) 

    # TRAIN MODELS
    base_path_best_models = "/media/na19/DATA/Naledi/denoising_data/Trained_models/DnCNNvs_dnet/"

    
    #save best model

    torch.save(dncnn_model.state_dict(), os.path.join(base_path_best_models, "DnCNN_model.pth"))
    torch.save(dnet.state_dict(), os.path.join(base_path_best_models, "dnet_model.pth"))

    # Optuna hyperparameter tuning >>> optuna samplers
    sampler = optuna.samplers.TPESampler()    
    study = optuna.create_study(sampler=sampler,direction='minimize')

    #study.optimize(func=objective, n_trials=2)
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, hyperparams), n_trials=10)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE] 
    if completed_trials:
        print("Best trial: ", study.best_trial)
   # print("Best trial: ", study.best_trial)
    # plot importance plots
    optuna.visualization.matplotlib.plot_param_importances(study)


    # plots of DnCNN vs. NoiseMapNet (dNet)

    epochs = range(hyperparams["num_epochs"])

    plt.figure(figsize=(10,6))
    plt.plot(epochs, all_losses["unet"]["train"], label="UNet Train Loss", linestyle='--')
    plt.plot(epochs, all_losses["unet"]["val"], label="UNet Val Loss", linestyle='-')
    plt.plot(epochs, all_losses["dncnn"]["train"], label="DnCNN Train Loss", linestyle='--')
    plt.plot(epochs, all_losses["dncnn"]["val"], label="DnCNN Val Loss", linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss: UNet vs DnCNN")
    plt.legend()
    plt.grid(True)
    plt.show()
    


if __name__ == "__main__":
    main()
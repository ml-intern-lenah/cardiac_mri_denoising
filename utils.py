#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 23:10:39 2025

@author: na19
"""

# Utility functions
from imports import *

# hyper parameters
def hyper_params():
    """
    Defining model hyper parameters
    
    Parameters:
    hyper_params (dict): defined model hyper-parameters
    
    Returns (dict): all the hyperparameters to be used
    """
    hyperparams = {
        "input_size": 16,
        "hidden_size": 128,
        "num_layers": 1,
        "num_classes": 1,
        "batch_size": 1,
        "num_epochs": 3, #100,#1000,
        "learning_rate": 1e-05,
        "adam_alpha_beta":(0.9, 0.999),
        "img_size":(64,64), # trained with (32,32)
        "noise_level":1000}
    
    return hyperparams


class Patch():
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    
    def patch(self, image, label, p, ps=None):
        """
        Obtains patched from full images
        
        Parameters:
        image (np.array): full size MRI image
        label (np.array): image label
        
        Returns (np.array): each patch and it's patch label
        """
        ps = self.patch_size if ps is None else ps
        
        # selecting the same patch for the image and the labels
        patch_img = image[p[0]:p[0] + ps[0], p[1]:p[1] + ps[1]]
        patch_lab = label[p[0]:p[0] + ps[0], p[1]:p[1] + ps[1]]
        
        return patch_img, patch_lab


class AllPatches(Patch):
    def __init__(self, patch_size):
        super().__init__(patch_size)
    
    
    def intensity_filter(self, image_patches):
        """
        Threshold the intensity for patch selection to avoid blank patches
        
        Parameters:
            image_patches (np.array): each indivially selected patch
            
        Returns (np.array): intensity thresholded output patch and label
    
        """
        intense = np.sum(np.reshape(image_patches, [image_patches.shape[0], -1]), axis=1) >12
        
        return intense


    def shuffle_patches(self, images, labels):
        """
        Randomises patch selection
        
        Parameters:
            image_patches (np.array):: each indivially selected patch and respective label
            
        Returns (np.array):: shuffled of patch-label pairs
        """
        
        ixs = np.arange(images.shape[0])
        ixs = np.random.permutation(ixs)
        
        return images[ixs], labels[ixs]


    def __call__(self, image, label):
        """
        Call to allow for the patches to be selected

        Parameters:
            image_patches (np.array): full-sized image and respective label
            
        Returns (np.array): selected shuffled and intensity-thresholded patch-label pairs
        """
        
        # choses patch of particular defined size
        new_size = np.floor(image.shape / np.array(self.patch_size)) * self.patch_size
        start = (image.shape - new_size) // 2
        image, label = self.patch(image, label, start.astype(np.int32), new_size.astype(np.int32))
        image_patches, label_patches = [view_as_blocks(x, tuple(self.patch_size)) for x in [image, label]]
        image_patches, label_patches = [np.reshape(x, [-1, *self.patch_size]) for x in [image_patches, label_patches]]
        
        # intensity filtering according to the image labels
        intense = self.intensity_filter(label_patches)
        image_patches, label_patches = image_patches[intense], label_patches[intense]
        
        return self.shuffle_patches(image_patches, label_patches)


def standardize_img(img, gauss_noise_input):
    """
    Standardises noisy image patch

    Parameters:
        img (tensor): torch input noisy image and it's added noise input
        gauss_noise_input(tensor): synthetically generated gaussian noise
        
    Returns (tensors): standardised noisy image and noise added to that image
    """
 
    #standardisation of image then paired noise input
    std, mean =  torch.std_mean(img, dim = (-2,-1),unbiased=False)
    std = std.reshape(std.shape + (1,1)).to(device)
    mean = mean.reshape(mean.shape + (1,1))
    
    std_noisy_image = (img - mean)/(std)
    std_gauss_noise = (gauss_noise_input.to(device)/(std))

    return std_noisy_image, std_gauss_noise, std


def convert_to_torchfloat(clean_img, std_noisy_img, std_gauss_noise):
    """
    Conversion of trensor data to float

    Parameters:
        clean_img (tensor): noise-free unstandardised image patch
        std_noisy_img(tensor): standardised image patch
        std_gauss_noise: standardised corresponding synthtically generated noise
        
    Returns (tensors -type floats): storch float32 clean and noisy standardised images
    """
    # moving data to device & setting them converting to float types
    clean_img, std_noisy_img  = clean_img.to(device), std_noisy_img.to(device)
    
    clean_img, std_noisy_img  = clean_img.to(torch.float32), std_noisy_img.to(torch.float32)
    std_gauss_noise = std_gauss_noise.to(torch.float32).to(device)
    
    return clean_img, std_noisy_img, std_gauss_noise



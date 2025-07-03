#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 23:40:34 2025

@author: na19
"""

from imports import *


class NumpyDataset(Dataset):
    # Dataset class
    
    def __init__(self, clean_image_folder, hyperparams, transform=None):
        
        # access data folders
        self.clean_image_folder = clean_image_folder
        self.imgs = list(sorted(os.listdir(self.clean_image_folder))) 
        
        self.hyperparams = hyperparams
        
        # data augmentations
        self.transform = transform

    def __len__(self):
        return  len(os.listdir(self.clean_image_folder))

    def __getitem__(self, idx):
        if idx >= len(self.imgs):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.imgs)}")
        img_path = os.path.join(self.clean_image_folder,self.imgs[idx])
        img_dcm=pydicom.dcmread(img_path)
        img = np.array(img_dcm.pixel_array).astype(np.float64)
    
        # resize/padding and patch selection
        desired_rows = 400
        desired_cols = 400
        new_clean_image = np.pad(img,((0, desired_rows - img.shape[0]), (0, desired_cols - img.shape[1])),
                                 'constant', constant_values=0)
        all_patches = AllPatches(self.hyperparams["img_size"])
        
        # patches selected
        new_clean_image, new_clean_image = all_patches(new_clean_image, new_clean_image)
        new_clean_image = torch.from_numpy(new_clean_image)
        
        return new_clean_image 


class NumpyDataset_non_mri(Dataset):
    # Non_MRI datasets
    def __init__(self, clean_image_folder, hyperparams, transform=None):
    
        # image folders
        self.clean_image_folder = clean_image_folder
        self.clean_image_sorted = os.listdir(self.clean_image_folder)
        self.clean_image_sorted.sort()
        
        self.hyperparams = hyperparams
        
        # tranforms for data augmentation
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.clean_image_folder))

    def __getitem__(self, idx):  
        # loading up the numpy images with their respective labels
        dcm_path = glob(self.clean_image_folder + "/*.png")
        
        for num in range(len(dcm_path)):
        
            # converting to numpy
            dcm_path=dcm_path[num]
            clean_image = imageio.imread(dcm_path)

            # zero padding the image to make them same size
            desired_rows = 200
            desired_cols = 200
            new_clean_image = clean_image;

            # selecting the image patches of size (x=64, y=64)
            all_patches = AllPatches(self.hyperparams["img_size"])
            new_clean_image, new_clean_image = all_patches(new_clean_image, new_clean_image)
        
            # if there is a transformation performed
            if self.transform:
                new_clean_image = self.transform(new_clean_image)
            
            # returning a certain batch of patches from the data (LABEL: CLEAN IMAGE, INPUT IMAGE: NOISY IMAGE)
            new_clean_image = torch.from_numpy(new_clean_image)
            
            return new_clean_image
    
    

#Reference:Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
from collections import OrderedDict
import subprocess
import json
import matplotlib.pyplot as plt
import nibabel as nib

import numpy as np
import torch
torch.set_num_threads(1)

import nnunetv2
print("Current Working Directory {}".format(os.getcwd()))
dir = "/Users/paula/PycharmProjects/masters_thesis/nnUNetFrame/dataset"
path_dict = {
    "nnUNet_raw" : os.path.join(dir, "nnUNet_raw"),
    "nnUNet_preprocessed" : os.path.join(dir, "nnUNet_preprocessed"), # 1 experiment: 1 epoch took 112s
    "nnUNet_results" : os.path.join(dir, "nnUNet_results"),
    "RAW_DATA_PATH" : os.path.join(dir, "RawData"), # This is used here only for convenience (not necessary for nnU-Net)!
}
import h5py
# Write paths to environment variables
for env_var, path in path_dict.items():
  os.environ[env_var] = path

# Check whether all environment variables are set correct!
for env_var, path in path_dict.items():
  if os.getenv(env_var) != path:
    print("Error:")
    print("Environment Variable {} is not set correctly!".format(env_var))
    print("Should be {}".format(path))
    print("Variable is {}".format(os.getenv(env_var)))

os.environ['NNUNET_PREPROCESSED'] = "C:\\Users\\paula\\PycharmProjects\\masters_thesis\\nnUNetFrame\\dataset\\nnUNet_preprocessed"
os.environ['NNUNET_RESULTS'] = "C:\\Users\\paula\\PycharmProjects\\masters_thesis\\nnUNetFrame\\dataset\\nnUNet_results"
os.environ['NNUNET_RAW'] = "C:\\Users\\paula\\PycharmProjects\\masters_thesis\\nnUNetFrame\\dataset\\nnUNet_raw"
print("NNUNET_PREPROCESSED:", os.getenv("NNUNET_PREPROCESSED"))
print("NNUNET_RESULTS:", os.getenv("NNUNET_RESULTS"))
print("NNUNET_RAW:", os.getenv("NNUNET_RAW"))
print("If No Error Occured Continue Forward. =)")


import stat

def set_permissions(path):
    # Walk through directory and set permissions for all files
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            os.chmod(filepath, stat.S_IWRITE)  # Set write permissions

def check_permissions(file_path):
    if os.access(file_path, os.R_OK) and os.access(file_path, os.W_OK):
        print(f"File {file_path} has read and write access.")
    else:
        print(f"File {file_path} is missing read/write access!")

# Check for specific .b2nd file permissions


# Call this function to ensure permissions are set properly
set_permissions('C:\\Users\\paula\\PycharmProjects\\masters_thesis\\nnUNetFrame\\dataset\\nnUNet_preprocessed')
set_permissions('C:\\Users\\paula\\PycharmProjects\\masters_thesis\\nnUNetFrame\\dataset\\nnUNet_results')
set_permissions('C:\\Users\\paula\\PycharmProjects\\masters_thesis\\nnUNetFrame\\dataset\\nnUNet_raw')
check_permissions('C:\\Users\\paula\\PycharmProjects\\masters_thesis\\nnUNetFrame\\dataset\\nnUNet_preprocessed\\Dataset001_Kaggle\\nnUNetPlans_3d_fullres\\patient_0000.b2nd')
#plots = visualization(train_image_dir, train_label_dir, test_image_dir, test_label_dir)
dataset_path = "/Users/paula/PycharmProjects/masters_thesis/nnUNetFrame/dataset/nnUNet_raw/Dataset001_Kaggle"

# Corrected command with absolute path
command = f'nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity --verbose'
# fold command is for crossvalidation setup
command1 = "nnUNetv2_train Dataset001_Kaggle 3d_fullres 0 -device cpu "

# Run the command
os.system(command1)


"""
Preprocess : nnUNetv2_plan_and_preprocess 001
For 2D: nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD

For 3D Full resolution: nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD

For Cascaded 3D:

First Run lowres: nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD

Then Run fullres: nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD"""


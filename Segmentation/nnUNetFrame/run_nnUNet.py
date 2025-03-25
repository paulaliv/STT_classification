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

os.environ['NNUNET_PREPROCESSED'] = "C:\\Users\\paula\\PycharmProjects\\STT_classification\\Segmentation\\nnUNetFrame\\nnUNet_preprocessed"
os.environ['NNUNET_RESULTS'] = "C:\\Users\\paula\\PycharmProjects\\STT_classification\\Segmentation\\nnUNetFrame\\nnUNet_results"
os.environ['NNUNET_RAW'] = "C:\\Users\\paula\\PycharmProjects\\STT_classification\\Segmentation\\nnUNetFrame\\nnUNet_raw"
print("NNUNET_PREPROCESSED:", os.getenv("NNUNET_PREPROCESSED"))
print("NNUNET_RESULTS:", os.getenv("NNUNET_RESULTS"))
print("NNUNET_RAW:", os.getenv("NNUNET_RAW"))


# Corrected command with absolute path
command = f'nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity --verbose'
# fold command is for crossvalidation setup
command1 = "nnUNetv2_train Dataset001_Kaggle 3d_fullres 0 -device cpu --npz "

command2 = "nnUNet_download_pretrained_model Task114_heart_mnms"




# Run the command
os.system(command2)


"""
Preprocess : nnUNetv2_plan_and_preprocess 001
For 2D: nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD

For 3D Full resolution: nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD

For Cascaded 3D:

First Run lowres: nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD

Then Run fullres: nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD"""

"""FIrst
 "nnUNet_train 3d_lowres nnUNetTrainerV2 TaskXXX_Folder fold0
 then:
    nnUNet_train 3d_cascade_fullres nnUNetTrainerV2 TaskXXX_Folder fold0
trained model will be: nnUNet_trained_models/nnUNet/3d_cascade_fullres/TaskXXX_Folder/nnUNetTrainerV2__nnUNetPlansv2.1/
nn
")

pip install hidden layer for pipeline"""
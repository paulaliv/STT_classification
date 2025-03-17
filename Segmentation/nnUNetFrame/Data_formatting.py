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

import h5py



def make_if_dont_exist(folder_path, overwrite=False):
    """
    creates a folder if it does not exists
    input:
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder
    """
    if os.path.exists(folder_path):

        if not overwrite:
            print(f"{folder_path} exists.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")

# Maybe move path of preprocessed data directly on content - this may be signifcantely faster!
print("Current Working Directory {}".format(os.getcwd()))
dir = "/Users/paula/PycharmProjects/masters_thesis/nnUNetFrame/dataset"
path_dict = {
    "nnUNet_raw" : os.path.join(dir, "nnUNet_raw"),
    "nnUNet_preprocessed" : os.path.join(dir, "nnUNet_preprocessed"), # 1 experiment: 1 epoch took 112s
    "nnUNet_results" : os.path.join(dir, "nnUNet_results"),
    "RAW_DATA_PATH" : os.path.join(dir, "RawData"), # This is used here only for convenience (not necessary for nnU-Net)!
}

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
  make_if_dont_exist(path, overwrite=False)

print("If No Error Occured Continue Forward. =)")

data_dir = "/Users/paula/PycharmProjects/masters_thesis/nnUNetFrame/dataset/lab_petct_vox_5.00mm.h5"

"""Convert data into nifti format, simple train/test split and save in correct directories"""
def data_conversion():
    with h5py.File(data_dir, "r") as f:
        patients = list(f["ct_data"].keys())

        # Create directories for train and test data
        train_image_dir = os.path.join(os.environ['nnUNet_raw'], 'imagesTr')
        train_label_dir = os.path.join(os.environ['nnUNet_raw'], 'labelsTr')
        test_image_dir = os.path.join(os.environ['nnUNet_raw'], 'imagesTs')
        test_label_dir = os.path.join(os.environ['nnUNet_raw'], 'labelsTs')

        make_if_dont_exist(train_image_dir, overwrite=True)
        make_if_dont_exist(train_label_dir, overwrite=True)
        make_if_dont_exist(test_image_dir, overwrite=True)
        make_if_dont_exist(test_label_dir, overwrite=True)


        for id in patients[:5]:
            # Convert CT image to NIfTI and save
            train_image = f["ct_data"][id][...]
            train_label = f["label_data"][id][...]

            train_image_nifti = nib.Nifti1Image(train_image, affine=np.eye(4))
            train_label_nifti = nib.Nifti1Image(train_label, affine=np.eye(4))

            # Save NIfTI files
            nib.save(train_image_nifti, os.path.join(train_image_dir, f"patient_{id}_ct.nii.gz"))
            nib.save(train_label_nifti, os.path.join(train_label_dir, f"patient_{id}_label.nii.gz"))

        # Save testing data (2 patients)
        for id in patients[5:]:
            # Access the image and label for patient i
            test_image = f["ct_data"][id][...]  # Read full dataset
            test_label = f["label_data"][id][...]

            test_image_nifti = nib.Nifti1Image(test_image, affine=np.eye(4))
            test_label_nifti = nib.Nifti1Image(test_label, affine=np.eye(4))

            # Save NIfTI files
            nib.save(test_image_nifti, os.path.join(test_image_dir, f"patient_{id}_ct.nii.gz"))
            nib.save(test_label_nifti, os.path.join(test_label_dir, f"patient_{id}_label.nii.gz"))


    print("Data conversion and saving complete!")
    return train_image_dir, train_label_dir, test_image_dir, test_label_dir

"""Creating the dataset.json file for data and pipeline fingerprint"""
def create_json(train_image_dir, test_image_dir):
    overwrite_json_file = True  # make it True if you want to overwrite the dataset.json file in Dataset_folder
    json_file_exist = False

    if os.path.exists(os.path.join(dir, 'dataset.json')):
        print('dataset.json already exist!')
        json_file_exist = True

    if json_file_exist == False or overwrite_json_file:

        json_dict = OrderedDict()
        json_dict['name'] = "Soft_tissue_tumours Kaggle"
        json_dict['description'] = "Soft Tissue Segmentation Kaggle"
        json_dict['tensorImageSize'] = "3D"
        #json_dict['reference'] = "see challenge website"
        #json_dict['licence'] = "see challenge website"
        #json_dict['release'] = "0.0"

        # you may mention more than one modality
        json_dict['channel_names'] = {
            "0": "MRI"
        }

        # set expected file ending
        json_dict["file_ending"] = ".nii.gz"

        # label names should be mentioned for all the labels in the dataset
        json_dict['labels'] = {
            "background": 0,
            "tumour": 1
        }

        train_ids = os.listdir(train_image_dir)
        test_ids = os.listdir(test_image_dir)
        json_dict['numTraining'] = len(train_ids)
        json_dict['numTest'] = len(test_ids)

        with open(os.path.join(dir, "dataset.json"), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)

        if os.path.exists(os.path.join(dir, 'dataset.json')):
            if json_file_exist == False:
                print('dataset.json created!')
            else:
                print('dataset.json overwritten!')

def visualization(train_image_dir, train_label_dir, test_image_dir, test_label_dir):
    train_image_name = os.listdir(train_image_dir)[2]
    train_label_name = train_image_name.replace('_ct.nii.gz', '_label.nii.gz')
    train_label = np.array(nib.load(os.path.join(train_label_dir, train_label_name)).dataobj)
    nonzero_slices = np.where(train_label.any(axis=(0, 1)))[0]
    if len(nonzero_slices) > 0:
        print(f"Tumor found in slices: {nonzero_slices}")
        middle = len(nonzero_slices) // 2
        start_slice = max(0, nonzero_slices[middle]-2)  # Show slices around the tumor
        end_slice = min(train_label.shape[2], start_slice + 5)

        train_img = np.array(nib.load(os.path.join(train_image_dir, train_image_name)).dataobj)[:, :, start_slice:end_slice]
        #print("Expected label file:", train_label_name)
        train_label = np.array(nib.load(os.path.join(train_label_dir, train_label_name)).dataobj)[:, :, start_slice:end_slice]

        print(train_img.shape, train_label.shape)
        print("Label unique values:", np.unique(train_label))

        max_rows = 2
        max_cols = train_img.shape[2]

        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
        for idx in range(max_cols):
            axes[0, idx].axis("off")
            axes[0, idx].set_title('Train Image' + str(idx + 1))
            axes[0, idx].imshow(train_img[:, :, idx], cmap="gray")
        for idx in range(max_cols):
            axes[1, idx].axis("off")
            axes[1, idx].set_title('Train Label' + str(idx + 1))
            axes[1, idx].imshow(train_label[:, :, idx])

        plt.subplots_adjust(wspace=.1, hspace=.1)
        plt.show()
    else:
        print("no tumour detected")


train_image_dir, train_label_dir, test_image_dir, test_label_dir = data_conversion()
#create_json(train_image_dir, test_image_dir)

#plots = visualization(train_image_dir, train_label_dir, test_image_dir, test_label_dir)
dataset_path = "/Users/paula/PycharmProjects/masters_thesis/nnUNetFrame/dataset/nnUNet_raw/Dataset001_Kaggle"

# Corrected command with absolute path
command = f'nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity --verbose'
command1 = "nnUNetv2_plan_and_preprocess -h"
# Run the command
os.system(command)

print("hello?")

"""
Preprocess : nnUNetv2_plan_and_preprocess 001
For 2D: nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD

For 3D Full resolution: nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD

For Cascaded 3D:

First Run lowres: nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD

Then Run fullres: nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD"""


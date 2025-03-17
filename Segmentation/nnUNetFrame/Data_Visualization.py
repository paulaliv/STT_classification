
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
from collections import OrderedDict

import json
import matplotlib.pyplot as plt
import nibabel as nib

import numpy as np
import torch
torch.set_num_threads(1)
#visualizing some of the training images and labels
# (re-run to see random pick-ups)
# only maximum of first 5 slices are plotted



def visualization(patient_id):
    dataset_path = "/Users/paula/PycharmProjects/masters_thesis/nnUNetFrame/dataset/nnUNet_raw/Dataset001_Kaggle"
    image_dir = f"{dataset_path}/imagesTr/patient_{patient_id}_0000.nii.gz/"
    label_dir = f"{dataset_path}/labelsTr/patient_{patient_id}.nii.gz/"
   # train_image_name = os.listdir(train_image_dir)[2]
    #train_label_name = train_image_name.replace('_ct.nii.gz', '_label.nii.gz')
    label = np.array(nib.load(label_dir).dataobj)
    nonzero_slices = np.where(label.any(axis=(0, 1)))[0]
    if len(nonzero_slices) > 0:
        print(f"Tumor found in slices: {nonzero_slices}")
        middle = len(nonzero_slices) // 2
        start_slice = max(0, nonzero_slices[middle]-2)  # Show slices around the tumor
        end_slice = min(label.shape[2], start_slice + 5)

        img = np.array(nib.load(os.path.join(image_dir)).dataobj)[:, :, start_slice:end_slice]
        #print("Expected label file:", train_label_name)
        label = np.array(nib.load(os.path.join(label_dir)).dataobj)[:, :, start_slice:end_slice]

        print(img.shape, label.shape)
        print("Label unique values:", np.unique(label))

        max_rows = 2
        max_cols = img.shape[2]

        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
        for idx in range(max_cols):
            axes[0, idx].axis("off")
            axes[0, idx].set_title('Train Image' + str(idx + 1))
            axes[0, idx].imshow(img[:, :, idx], cmap="gray")
        for idx in range(max_cols):
            axes[1, idx].axis("off")
            axes[1, idx].set_title('Train Label' + str(idx + 1))
            axes[1, idx].imshow(label[:, :, idx])

        plt.subplots_adjust(wspace=.1, hspace=.1)
        plt.show()
    else:
        print("no tumour detected")


if __name__ == "__main__":
    patient_id = "0002"
    visualization(patient_id)

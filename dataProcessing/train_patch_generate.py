import os
import numpy as np
import SimpleITK as sitk
import natsort as nt
import random
from batchgenerators.augmentations.spatial_transformations import augment_mirroring


def train_patch_generate(ct_arr, seg_arr, flag, x, y, z):
    x_max = ct_arr.shape[0]
    y_max = ct_arr.shape[1]
    z_max = ct_arr.shape[2]
    if flag:
        while True:
            temp = np.nonzero(seg_arr)
            center = random.randint(0, temp[0].shape[0]-1)
            center_x = temp[0][center]
            center_y = temp[1][center]
            center_z = temp[2][center]
            start_x, end_x = int(center_x - x/2), int(center_x + x/2)
            start_y, end_y = int(center_y - y/2), int(center_y + y/2)
            start_z, end_z = int(center_z - z/2), int(center_z + z/2)
            if start_x < 0 or end_x > x_max or start_y < 0 or end_y > y_max or start_z < 0 or end_z > z_max:
                continue
            else:
                if np.random.uniform() < 0.5714:
                    img_result = ct_arr[start_x:end_x, start_y:end_y, start_z:end_z]
                    seg_result = seg_arr[start_x:end_x, start_y:end_y, start_z:end_z]
                    img_result = img_result[None]
                    seg_result = seg_result[None]
                    img_result, seg_result = augment_mirroring(img_result, seg_result)
                    return img_result[0, :, :, :], seg_result[0, :, :, :]
                else:
                    return ct_arr[start_x:end_x, start_y:end_y, start_z:end_z], seg_arr[start_x:end_x, start_y:end_y, start_z:end_z]
    else:
        start_x = random.randint(0, x_max-x-2)
        end_x = start_x + x
        start_y = random.randint(0, y_max-y-2)
        end_y = start_y + y
        start_z = random.randint(0, z_max-z-2)
        end_z = start_z + z
        if np.random.uniform() < 0.5714:
            img_result = ct_arr[start_x:end_x, start_y:end_y, start_z:end_z]
            seg_result = seg_arr[start_x:end_x, start_y:end_y, start_z:end_z]
            img_result = img_result[None]
            seg_result = seg_result[None]
            img_result, seg_result = augment_mirroring(img_result, seg_result)
            return img_result[0, :, :, :], seg_result[0, :, :, :]
        else:
            return ct_arr[start_x:end_x, start_y:end_y, start_z:end_z], seg_arr[start_x:end_x, start_y:end_y, start_z:end_z]


if __name__ == '__main__':
    img_path = r'/media/image522/221D883AB7DA6CE4/qcj/3DPatchSegmentation/dataset1/image/pancreas0001.nii.gz'
    label_path = r'/media/image522/221D883AB7DA6CE4/qcj/3DPatchSegmentation/dataset1/label/label0001.nii.gz'
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)
    label = sitk.ReadImage(label_path)
    label_arr = sitk.GetArrayFromImage(label)
    ct, seg = train_patch_generate(img_arr, label_arr, 0, 96, 160, 96)
    print('ct.shape:', ct.shape)
    print('seg.shape:', seg.shape)
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from dataProcessing.train_patch_generate import train_patch_generate
import natsort as nt
import parameters as para
import os
from batchgenerators.augmentations.color_augmentations import augment_brightness_multiplicative, augment_contrast


class Dataset(dataset):
    def __init__(self, image_dir):
        self.img_dir = os.path.join(image_dir, 'image')
        self.ct_list, self.seg_list = self.get_train_list()
        # print(self.ct_list)
        # print(self.seg_list)

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # z-score
        ct_array = ct_array.astype(np.float32)
        ct_array[ct_array < para.lower] = para.lower
        ct_array[ct_array > para.upper] = para.upper
        mean = np.mean(ct_array)
        std = np.std(ct_array)
        ct_array = (ct_array - mean) / std
        if np.random.uniform() <= 0.15:
            ct_array = augment_brightness_multiplicative(ct_array, multiplier_range=(0.7, 1.3), per_channel=False)
        if np.random.uniform() <= 0.15:
            ct_array = augment_contrast(ct_array, (0.75, 1.25), per_channel=False)
        ran_num = random.randint(1, 10)
        if ran_num <= 7:
            flag = 1
            ct_array, seg_array = train_patch_generate(ct_array, seg_array, flag, para.patch_size[0], para.patch_size[1], para.patch_size[2])
        if ran_num > 7:
            flag = 0
            ct_array, seg_array = train_patch_generate(ct_array, seg_array, flag, para.patch_size[0], para.patch_size[1], para.patch_size[2])

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):
        return len(self.ct_list)

    def get_train_list(self):
        ct_list = []
        seg_list = []
        for img in nt.natsorted(os.listdir(self.img_dir)):
            ct_list.append(os.path.join(self.img_dir, img))
            seg_list.append(os.path.join(self.img_dir, img).replace(r'/image/', r'/label/').replace(r'pancreas0', r'label0'))
        return ct_list, seg_list


if __name__ == '__main__':
    img_dir = r'/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset/test'
    object1 = Dataset(img_dir)
    print(object1.__len__())


import os
import shutil
import natsort as nt
import parameters as para
import random


def create_train_val_test_data(root_path=None):  # train:val:test = 7:1:2
    img_path = os.path.join(root_path, 'image')
    label_path = os.path.join(root_path, 'label')
    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'val')
    test_path = os.path.join(root_path, 'test')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    data_index = nt.natsorted(list(range(1, 83)))
    random.seed(1)
    train_index = nt.natsorted(random.sample(data_index, 57))
    test_val_index = nt.natsorted(list(set(data_index) - set(train_index)))
    random.seed(0)
    val_index = nt.natsorted(random.sample(test_val_index, 8))
    test_index = nt.natsorted(list(set(test_val_index) - set(val_index)))
    train_img_path = os.path.join(train_path, 'image')
    train_label_path = os.path.join(train_path, 'label')
    val_img_path = os.path.join(val_path, 'image')
    val_label_path = os.path.join(val_path, 'label')
    test_img_path = os.path.join(test_path, 'image')
    test_label_path = os.path.join(test_path, 'label')
    if not os.path.exists(train_img_path):
        os.mkdir(train_img_path)
    if not os.path.exists(train_label_path):
        os.mkdir(train_label_path)
    if not os.path.exists(val_img_path):
        os.mkdir(val_img_path)
    if not os.path.exists(val_label_path):
        os.mkdir(val_label_path)
    if not os.path.exists(test_img_path):
        os.mkdir(test_img_path)
    if not os.path.exists(test_label_path):
        os.mkdir(test_label_path)

    for i in nt.natsorted(train_index):
        train_img_path_src = os.path.join(img_path, 'pancreas' + str(i).zfill(4)+'.nii.gz')
        train_label_path_src = os.path.join(label_path, 'label' + str(i).zfill(4) + '.nii.gz')
        train_img_path_dst = train_img_path
        train_label_path_dst = train_label_path
        shutil.copy(train_img_path_src, train_img_path_dst)
        shutil.copy(train_label_path_src, train_label_path_dst)

    for i in nt.natsorted(val_index):
        val_img_path_src = os.path.join(img_path, 'pancreas' + str(i).zfill(4)+'.nii.gz')
        val_label_path_src = os.path.join(label_path, 'label' + str(i).zfill(4) + '.nii.gz')
        val_img_path_dst = val_img_path
        val_label_path_dst = val_label_path
        shutil.copy(val_img_path_src, val_img_path_dst)
        shutil.copy(val_label_path_src, val_label_path_dst)

    for i in nt.natsorted(test_index):
        test_img_path_src = os.path.join(img_path, 'pancreas' + str(i).zfill(4)+'.nii.gz')
        test_label_path_src = os.path.join(label_path, 'label' + str(i).zfill(4) + '.nii.gz')
        test_img_path_dst = test_img_path
        test_label_path_dst = test_label_path
        shutil.copy(test_img_path_src, test_img_path_dst)
        shutil.copy(test_label_path_src, test_label_path_dst)


if __name__ == '__main__':
    create_train_val_test_data(para.root_path)

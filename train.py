import os
from time import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import Dataset
from loss.Dice import DiceLoss
from loss.BCE import BCELoss
from net.PAHRNet3D import net
import SimpleITK as sitk
import parameters as para
from utilities.calculate_metrics import Metirc
import skimage.morphology as morphology
import skimage.measure as measure
from dataProcessing.test_patch_generate import test_seg_generate
import numpy as np
import copy

# 设置显卡相关
# gpu = '0'
# cudnn_benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu  # 指定使用那几块gpu,多块可以用逗号隔开
# 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个
# 网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），
# 网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，
# 如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。pytorch默认为false
cudnn.benchmark = para.cudnn_benchmark
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(network):
    if not os.path.exists(para.model_save_path):
        os.mkdir(para.model_save_path)
    # 定义网络
    # network.load_state_dict(torch.load(para.model_read_path))
    network = network.to(device)
    network.train()

    # 定义Dateset
    train_ds = Dataset(para.train_path)

    # 定义数据加载
    # pin_memory默认为False,当为True从Cpu转到Gpu会变快，但更占内存，选择true时，机器要求高点
    train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers,
                          pin_memory=para.pin_memory)  # pin_memory = True
    # print(train_dl.__len__())
    # 挑选损失函数
    loss_func_list = [DiceLoss(), BCELoss()]
    dice_loss_func = loss_func_list[0]
    bce_loss_func = loss_func_list[1]

    # 定义优化器
    opt = torch.optim.Adam(network.parameters(), lr=para.learning_rate)
    # 学习率衰减
    # # learning_rate_decay = [600, 750]
    # # 小于600epoch，lr=lr；600到750之间, lr=0.1lr;大于750, lr=0.01lr.lr为初始学习率
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

    # 训练网络
    start = time()
    for epoch in range(para.Epoch):
        mean_loss = []

        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.to(device)
            seg = seg.to(device)
            outputs = network(ct)
            loss = (dice_loss_func(outputs, seg) + bce_loss_func(outputs, seg)) / 2
            mean_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 5 == 0:
                #   step_list.append(step_list[-1] + 1)
                # viz.line(X=np.array([step_list[-1]]), Y=np.array([loss4.item()]), win=win, update='append')

                # print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                # .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                      .format(epoch, step, loss.item(), (time() - start) / 60))
                model_save_path = os.path.join(para.model_save_path, 'train.txt')
                with open(model_save_path, 'a') as f:
                    f.write('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                            .format(epoch, step, loss.item(), (time() - start) / 60))
                    f.write('\r\n')

        lr_decay.step()
        mean_loss = sum(mean_loss) / len(mean_loss)

        # 保存模型
        if epoch % 50 == 0 and epoch != 0:
            # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
            torch.save(network.state_dict(), os.path.join(para.model_save_path,
                                                      'network{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss)))
            val(network, epoch)


def val(network, epoch):
    # 为了计算dice_global定义的两个变量
    dice_intersection = 0.0
    dice_union = 0.0
    network.eval()
    val_ct_path = os.path.join(para.val_path, r'image')
    with torch.no_grad():
        for file_index, file in enumerate(os.listdir(val_ct_path)):
            start = time()
            # 将CT读入内存
            ct = sitk.ReadImage(os.path.join(val_ct_path, file), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)

            origin_shape = ct_array.shape
            # 将灰度值在阈值之外的截断掉
            ct_array[ct_array > para.upper] = para.upper
            ct_array[ct_array < para.lower] = para.lower

            # z-score
            ct_array = ct_array.astype(np.float32)
            ct_array = (ct_array - np.mean(ct_array)) / np.std(ct_array)

            # 将金标准读入内存
            seg = sitk.ReadImage(os.path.join(val_ct_path, file.replace('pancreas0', 'label0'))
                                 .replace(r'/image/', r'/label/'), sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            pred_seg = test_seg_generate(ct_array, device, network, True)
            # 对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充
            pred_seg = pred_seg.astype(np.uint8)
            pancreas_seg = copy.deepcopy(pred_seg)
            pancreas_seg = measure.label(pancreas_seg, connectivity=1)
            props = measure.regionprops(pancreas_seg)

            max_area = 0
            max_index = 0
            for index, prop in enumerate(props, start=1):
                if prop.area > max_area:
                    max_area = prop.area
                    max_index = index

            pancreas_seg[pancreas_seg != max_index] = 0
            pancreas_seg[pancreas_seg == max_index] = 1

            pancreas_seg = pancreas_seg.astype(np.bool_)
            morphology.remove_small_holes(pancreas_seg, para.maximum_hole, connectivity=2, in_place=True)
            pancreas_seg = pancreas_seg.astype(np.uint8)

            # 计算分割评价指标
            pancreas_metric = Metirc(seg_array, pancreas_seg, ct.GetSpacing())
            dice_intersection += pancreas_metric.get_dice_coefficient()[1]
            dice_union += pancreas_metric.get_dice_coefficient()[2]
            speed = time() - start
            print(file_index, 'this case use {:.3f} s'.format(speed))
            print('-----------------------')
        # 打印dice global
        print('dice global:', dice_intersection / dice_union)
        temp_path = os.path.join(para.model_save_path, 'val_log.txt')
        with open(temp_path, 'a+') as f:
            f.write('epoch ' + str(epoch) + '_' + 'dice global:' + str(dice_intersection / dice_union))
            f.write('\n')


if __name__ == '__main__':
    train(net)

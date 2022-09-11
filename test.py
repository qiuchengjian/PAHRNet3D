import os
import copy
import collections
from time import time
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology
from net.PAHRNet3D import net
from utilities.calculate_metrics import Metirc
from dataProcessing.test_patch_generate import test_seg_generate
import parameters as para


os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 为了计算dice_global定义的两个变量
dice_intersection = 0.0
dice_union = 0.0

file_name = []  # 文件名称
time_per_case = []  # 单例数据消耗时间

# 定义评价指标
pancreas_score = collections.OrderedDict()
pancreas_score['dice'] = []
pancreas_score['jacard'] = []
# pancreas_score['voe'] = []
pancreas_score['fnr'] = []
pancreas_score['fpr'] = []
pancreas_score['recall'] = []
pancreas_score['precision'] = []
# pancreas_score['assd'] = []
# pancreas_score['rmsd'] = []
# pancreas_score['msd'] = []
# 定义网络并加载参数
net = net.to(device)
net.load_state_dict(torch.load(para.model_read_path))
net.eval()

test_ct_path = os.path.join(para.test_path, 'image')
if not os.path.exists(para.pred_path):
    os.mkdir(para.pred_path)

for file_index, file in enumerate(os.listdir(test_ct_path)):

    start = time()

    file_name.append(file)

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape
    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # z-score
    ct_array = ct_array.astype(np.float32)
    ct_array = (ct_array-np.mean(ct_array)) / np.std(ct_array)

    # 将金标准读入内存
    seg = sitk.ReadImage(os.path.join(test_ct_path, file.replace('pancreas0', 'label0'))
                         .replace(r'/image/', r'/label/'), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    pred_seg = test_seg_generate(ct_array, device, net, False)
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

    pancreas_score['dice'].append(pancreas_metric.get_dice_coefficient()[0])
    pancreas_score['jacard'].append(pancreas_metric.get_jaccard_index())
    # pancreas_score['voe'].append(pancreas_metric.get_VOE())
    pancreas_score['fnr'].append(pancreas_metric.get_FNR())
    pancreas_score['fpr'].append(pancreas_metric.get_FPR())
    pancreas_score['recall'].append(pancreas_metric.get_recall())
    pancreas_score['precision'].append(pancreas_metric.get_precision())
    # pancreas_score['assd'].append(pancreas_metric.get_ASSD())
    # pancreas_score['rmsd'].append(pancreas_metric.get_RMSD())
    # pancreas_score['msd'].append(pancreas_metric.get_MSD())

    dice_intersection += pancreas_metric.get_dice_coefficient()[1]
    dice_union += pancreas_metric.get_dice_coefficient()[2]

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pancreas_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.pred_path, file.replace('pancreas0', 'pred0')))

    speed = time() - start
    time_per_case.append(speed)

    print(file_index, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')


# 将评价指标写入到exel中
pancreas_data = pd.DataFrame(pancreas_score, index=file_name)
pancreas_data['time'] = time_per_case

pancreas_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(pancreas_data.columns))
pancreas_statistics.loc['mean'] = pancreas_data.mean()
pancreas_statistics.loc['std'] = pancreas_data.std()
pancreas_statistics.loc['min'] = pancreas_data.min()
pancreas_statistics.loc['max'] = pancreas_data.max()

result_path = os.path.join(para.pred_path, 'result.xlsx')
writer = pd.ExcelWriter(result_path)
pancreas_data.to_excel(writer, 'pancreas')
pancreas_statistics.to_excel(writer, 'pancreas_statistics')
writer.save()

# 打印dice global
print('dice global:', dice_intersection / dice_union)

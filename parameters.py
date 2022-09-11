# -----------------------路径相关参数---------------------------------------
root_path = r'/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset'

train_path = r'/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset/train'

val_path = r'/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset/val'

test_path = r'/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset/test'

pred_path = r'/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset/pred'  # 网络预测结果保存路径

model_save_path = r'/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset/model'

model_read_path = '/media/image522/221D883AB7DA6CE4/qcj/HRNet_related/dataset/model/network800-0.042-0.193.pth'  # 测试模型地址


# -----------------------路径相关参数---------------------------------------


# ---------------------训练数据获取相关参数-----------------------------------

upper, lower = 240., -100.  # CT数据灰度截断窗口

k_folder = 4  # 第4折， 即前3折训练，第4折测试
n_folder = 4  # 使用4折交叉验证

# ---------------------网络训练相关参数--------------------------------------

gpu = '0'  # 使用的显卡序号

Epoch = 1001
learning_rate = 1e-4

learning_rate_decay = [600, 800]

patch_size = 96, 160, 96

batch_size = 1

num_workers = 3

pin_memory = True

cudnn_benchmark = True

# ---------------------网络训练相关参数--------------------------------------


# ----------------------模型测试相关参数-------------------------------------

threshold = 0.5  # 阈值度阈值

val_stride_x, val_stride_y, val_stride_z = 48, 80, 48

stride_x, stride_y, stride_z = 32, 64, 32  # 滑动取样步长

maximum_hole = 5e4  # 最大的空洞面积


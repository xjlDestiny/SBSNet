
import os
from time import time

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

from models.MSSAE import EncoderBlock, DecoderBlock
from utils import maxMinNormalization, tsGeneration, spectralSimilarity, cornerPointCalculation, \
    experimentalResultsDisplay

from setting import *

device = torch.device('cuda:2')
# device = torch.device('cpu')


encoder_path = 'XXXXX'


""" 数据加载 """
window_size = 5
# window_size = 3
HSI_DATA_PATH = "dataset/AVIRIS/AVIRIS-I.npy"
HSI_GT_PATH = "dataset/AVIRIS/AVIRIS-I-gt.npy"
# HSI_GT_PATH = "dataset/Abu-airport/Abu-airport-2-gt.npy"
# HSI_DATA_PATH = "dataset/Abu-airport/Abu-airport-2.npy"
# HSI_DATA_PATH = "dataset/Abu-beach/abu-beach-2.npy"
# HSI_GT_PATH = "dataset/Abu-beach/abu-beach-2-gt.npy"
# HSI_DATA_PATH = "dataset/Abu-urban/abu-urban-2.npy"
# HSI_GT_PATH = "dataset/Abu-urban/abu-urban-2-gt.npy"
# HSI_GT_PATH = "dataset/Qingpu/Qingpu-I-gt.npy"
# HSI_DATA_PATH = "dataset/Qingpu/Qingpu-I.npy"
HSI_DATA_NAME = os.path.splitext(os.path.basename(HSI_DATA_PATH))[0]
hsi_origin = np.load(HSI_DATA_PATH)
gt = np.load(HSI_GT_PATH)
print(gt.sum())
hsi_origin = maxMinNormalization(hsi_origin)
ts = tsGeneration(hsi_origin, gt, type_target_spectrum)

# 初始化模型
in_len = ts.shape[-1]
encoder = EncoderBlock(in_len, mapped_len, d_input, d_model, layers=layers,
            ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers, device=device)
decoder = DecoderBlock(mapped_len, in_len, d_model, d_input, use_unpooling=False, layers=layers,
            ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers, device=device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
summary(encoder, (1, in_len), device=device)
save_dir = os.path.dirname(encoder_path)



start_time = time()

# test data
hsi_w = cornerPointCalculation(hsi_origin, window_size=window_size, metric='SA')
hsi_list = [hsi_origin, hsi_w]
hsi_t_list = []
hsi_b_list = []
hsi = None
for hsi_i in hsi_list:
    hsi_t_list.append(torch.from_numpy(hsi_i[np.where(gt == 1)]))
    hsi_b_list.append(torch.from_numpy(hsi_i[np.where(gt == 0)]))
    if hsi is None:
        hsi = hsi_i
    else:
        hsi += hsi_i
hsi = hsi / len(hsi_list)
ts_tensor = torch.from_numpy(ts)
hsi_tensor = torch.tensor(hsi.reshape(-1, hsi.shape[-1]), dtype=torch.float32, device=device)
label_tensor = torch.tensor(gt.reshape(-1), dtype=torch.int8, device=device)
test_dataset = TensorDataset(hsi_tensor, label_tensor)
test_iter = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

with torch.no_grad():  # 确保不进行梯度计算以节省内存

    outputs = []  # 创建一个列表来存储每一次的输出 Tensor
    encoder.eval()

    from thop import profile
    # 构造 dummy 输入：必须与 prior_target 输入到 embed_net 的形状一致
    dummy_input = torch.randn_like(ts_tensor).to(device)
    flops, params = profile(encoder, inputs=(dummy_input,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Params: {params / 1e6:.4f} M parameters")

    for x, y in test_iter:
        x = x.to(device)
        _, mapped_x = encoder(x)
        outputs.append(mapped_x)  # 将每一次的输出添加到 outputs 列表中
    mapped_hsi = torch.cat(outputs, dim=0)  # 在第 0 维上拼接所有的输出 Tensor
    ts_tensor = ts_tensor.to(device)
    _, mapped_ts = encoder(ts_tensor)
# result = CEM(final_output.squeeze(), ts_tensor)
# detection_map = result.reshape(*gt.shape)
similarity = spectralSimilarity(mapped_hsi.squeeze(), mapped_ts, metric='SA')
detection_map = similarity.reshape(*gt.shape)
end_time = time()
print('testing time:', end_time - start_time)

# with torch.no_grad():  # 确保不进行梯度计算以节省内存
#     print(device)
#     outputs = []  # 创建一个列表来存储每一次的输出 Tensor
#     encoder.eval()
#     for x, y in test_iter:
#         x = x.to(device)
#         ts_tensor = ts_tensor.to(device)
#         _, mapped_x = encoder(x)
#         outputs.append(mapped_x)  # 将每一次的输出添加到 outputs 列表中
#     mapped_hsi = torch.cat(outputs, dim=0)  # 在第 0 维上拼接所有的输出 Tensor
#     _, mapped_ts = encoder(ts_tensor)
# similarity = spectralSimilarity(mapped_hsi.squeeze(), mapped_ts, metric='SA')
# detection_map = similarity.reshape(*gt.shape)
#
# auc = experimentalResultsDisplay(hsi, gt, np.array(detection_map.cpu()))
# print(auc)


detection_map = np.array(detection_map.cpu())
y_l = np.reshape(gt, [-1, 1])  # 'F' 代表 Fortran 风格 (列优先, 或者称为按列主序)
y_p = detection_map.reshape([-1, 1])
fpr, tpr, thresholds = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
# 在计算AUC时，ROC曲线的起点(0, 0)并不被考虑
fpr = fpr[1:]
tpr = tpr[1:]
thresholds = thresholds[1:]
auc_ft = round(metrics.auc(fpr, tpr), 5)
auc_t = round(metrics.auc(thresholds, tpr), 5)
auc_f = round(metrics.auc(thresholds, fpr), 5)
auc_oa = round(auc_ft + auc_t - auc_f, 5)
auc_snpr = round(auc_t / auc_f, 5)
print('auc_ft: {:.{precision}f}'.format(auc_ft, precision=5))
print('auc_t: {:.{precision}f}'.format(auc_t, precision=5))
print('auc_f: {:.{precision}f}'.format(auc_f, precision=5))
print('auc_oa: {:.{precision}f}'.format(auc_oa, precision=5))
print('auc_snpr: {:.{precision}f}'.format(auc_snpr, precision=5))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(detection_map)
plt.axis('off')
plt.savefig(os.path.join(save_dir, '{}_{}_detection_map.png'.format('BTSNet', HSI_DATA_NAME)), bbox_inches='tight')
plt.close()
np.save(os.path.join(save_dir, '{}_{}_detection_map.npy'.format('BTSNet', HSI_DATA_NAME)), detection_map)

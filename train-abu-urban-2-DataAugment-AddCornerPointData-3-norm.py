

import os

import numpy as np
import torch

from Train_Test_norm import train, dataAugment
from models import detector
from utils import maxMinNormalization, cornerPointCalculation, tsGeneration, calculation_Binary_map

from setting import *


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(device)


""" 数据加载 """
HSI_GT_PATH = "dataset/Abu-urban/abu-urban-2-gt.npy"
HSI_DATA_PATH = "dataset/Abu-urban/abu-urban-2.npy"
HSI_DATA_NAME = os.path.splitext(os.path.basename(HSI_DATA_PATH))[0]
hsi_origin = np.load(HSI_DATA_PATH)
gt = np.load(HSI_GT_PATH)
print(gt.sum())
hsi_origin = maxMinNormalization(hsi_origin)
ts = tsGeneration(hsi_origin, gt, type_target_spectrum)
window_size = 3
hsi_w = cornerPointCalculation(hsi_origin, window_size=window_size, metric='SA')

# hsi_list = [hsi_origin.reshape(-1, hsi_origin.shape[-1]), hsi_w3.reshape(-1, hsi_w3.shape[-1])]
hsi_list = [hsi_origin.reshape(-1, hsi_origin.shape[-1]), hsi_w.reshape(-1, hsi_w.shape[-1])]
hsi = None
for hsi_i in hsi_list:
    if hsi is None:
        hsi = hsi_i
    else:
        hsi += hsi_i
hsi = hsi / len(hsi_list)
hsi_list.append(hsi)

ACE = detector.ACE()
detection_map_ace = ACE(hsi, ts)
detection_map_ace = maxMinNormalization(detection_map_ace)
Binary_map, Binary_map_median_filter = calculation_Binary_map(
    detection_map_ace, gt, folder_display="./abu-urban-2/{}/".format(window_size))
pos_t = [tuple(index) for index in np.argwhere(Binary_map == 1)]
pos_b = [tuple(index) for index in np.argwhere(Binary_map == 0)]
hsi_t_list = []
hsi_b_list = []
for hsi_i in hsi_list:
    # 使用排序后的索引对hsi_t_tensor中的光谱向量进行排序
    hsi_i_tensor = torch.from_numpy(hsi_i)
    # hsi_t_list.append(hsi_i_tensor[Binary_map == 1])
    # hsi_b_list.append(hsi_i_tensor[Binary_map == 0])
    hsi_t_list.append(hsi_i_tensor[pos_t].squeeze())
    hsi_b_list.append(hsi_i_tensor[pos_b].squeeze())
hsi_b_origin = torch.cat(hsi_b_list, dim=0)
hsi_t_origin = torch.cat(hsi_t_list, dim=0)

for seed in seeds:
    alaf = 0.2
    # 设置随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    # 定义保存路劲
    current_file_name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    save_dir = "log/{}/{}/dataAugment/".format(current_file_name, MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)
    log_path = "log/{}/{}/{}--dModel-{}_layers-{}_ffnHidden-{}_nHead-{}_nLayers-{}--{}-{}-{}-{}-{}-{}-{}-{}/".format(
        current_file_name, MODEL_NAME, type_target_spectrum, d_model,
        layers, ffn_hidden, n_head, n_layers, weight_decay, gamma, encode_f,
        master_seed, seed, alaf, epochs, step_decay)
    os.makedirs(log_path, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, 'encoder.pth')):
        hsi_t_augment_tensor = torch.from_numpy(np.load(os.path.join(
            save_dir, '{}_{}_hsi_t_all_augmented.npy'.format(len(hsi_list), window_size))))
        hsi_b_augment_tensor = torch.from_numpy(np.load(os.path.join(
            save_dir, '{}_{}_hsi_b_all_augmented.npy'.format(len(hsi_list), window_size))))
        # hsi_t_augment_tensor = torch.cat([
        #     torch.from_numpy(np.load(os.path.join(save_dir, '3_hsi_t_all_augmented_29754.npy'))),
        #     torch.from_numpy(np.load(os.path.join(save_dir, 'hsi_t_all_augmented_119016.npy')))
        # ])
        # hsi_b_augment_tensor = torch.from_numpy(np.load(os.path.join(save_dir, 'hsi_b_all_augmented_118956.npy')))
    else:
        hsi_t_augment_tensor = dataAugment(hsi_t_origin, num_augment=len(hsi_b_origin) * 5, batch_size=32,
                                           num_epochs=5000000, save_dir=save_dir, device=device, type='t',
                                           num_cluster='all', num_data=len(hsi_list), window_size=window_size)
        hsi_b_augment_tensor = dataAugment(hsi_b_origin, num_augment=len(hsi_b_origin) * 4, batch_size=128,
                                           num_epochs=5000000, save_dir=save_dir, device=device, type='b',
                                           num_cluster='all', num_data=len(hsi_list), window_size=window_size)
    hsi_b_tensor = torch.cat([hsi_b_origin, hsi_b_augment_tensor], dim=0)
    hsi_t_tensor = torch.cat([hsi_t_origin, hsi_t_augment_tensor], dim=0)
    ts_tensor = torch.from_numpy(ts)


    train(hsi, gt, HSI_DATA_NAME, hsi_b_tensor, hsi_t_tensor,
          ts_tensor, log_path, save_dir, device, alaf=alaf, Binary_map=Binary_map.reshape(*gt.shape))
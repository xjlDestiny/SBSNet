
import os

import torch

from Train_Test_norm import dataAugment
from utils import maxMinNormalization, tsGeneration, spectralSimilarity, cornerPointCalculation

import numpy as np



if __name__ == "__main__":

    """ 数据加载 """
    # HSI_GT_PATH = "dataset/Abu-airport/Abu-airport-2-gt.npy"
    # HSI_DATA_PATH = "dataset/Abu-airport/Abu-airport-2.npy"
    # HSI_GT_PATH = "dataset/Abu-beach/abu-beach-2-gt.npy"
    # HSI_DATA_PATH = "dataset/Abu-beach/abu-beach-2.npy"
    HSI_GT_PATH = "dataset/Abu-urban/abu-urban-2-gt.npy"
    HSI_DATA_PATH = "dataset/Abu-urban/abu-urban-2.npy"
    # HSI_GT_PATH = "dataset/AVIRIS/AVIRIS-I-gt.npy"
    # HSI_DATA_PATH = "dataset/AVIRIS/AVIRIS-I.npy"
    # HSI_GT_PATH = "dataset/AVIRIS/AVIRIS-II-gt.npy"
    # HSI_DATA_PATH = "dataset/AVIRIS/AVIRIS-II.npy"
    # HSI_GT_PATH = "dataset/Sandiego-gt.npy"
    # HSI_DATA_PATH = "dataset/Sandiego.npy"
    type_target_spectrum = 2
    window_size = 7
    hsi_origin = np.load(HSI_DATA_PATH)
    gt = np.load(HSI_GT_PATH)
    hsi_origin = maxMinNormalization(hsi_origin)
    ts = tsGeneration(hsi_origin, gt, type_target_spectrum)
    hsi_w = cornerPointCalculation(hsi_origin, window_size=window_size, metric='SA')
    hsi_list = [hsi_origin, hsi_w]
    hsi = None
    for hsi_i in hsi_list:
        if hsi is None:
            hsi = hsi_i
        else:
            hsi += hsi_i
    hsi = hsi / len(hsi_list)
    hsi_list.append(hsi)
    # 获取当前文件的名字
    current_file = os.path.basename(__file__)
    # 通过os.path.splitext()方法去掉扩展名，获取文件名
    file_name_without_ext = os.path.splitext(current_file)[0]
    HSI_DATA_NAME = os.path.splitext(os.path.basename(HSI_DATA_PATH))[0]
    save_dir = 'log/{}/{}/'.format(file_name_without_ext, HSI_DATA_NAME)
    os.makedirs(save_dir, exist_ok=True)
    hsi_t_list = []
    hsi_b_list = []
    for data in hsi_list:
        hsi_t_list.append(torch.from_numpy(data[np.where(gt == 1)]))
        hsi_b_list.append(torch.from_numpy(data[np.where(gt == 0)]))
    hsi_b_tensor = torch.cat(hsi_b_list, dim=0)
    hsi_t_tensor = torch.cat(hsi_t_list, dim=0)
    ts_tensor = torch.from_numpy(ts)
    print(hsi_b_tensor.shape)
    print(hsi_t_tensor.shape)

    hsi_t_similarity = spectralSimilarity(hsi_t_tensor, ts_tensor, metric='SA')
    # print(hsi_t_similarity)
    # print(hsi_t_similarity.max(), hsi_t_similarity.min())
    hsi_b_similarity = spectralSimilarity(hsi_b_tensor, ts_tensor, metric='SA')
    # print(hsi_b_similarity)
    # print(hsi_b_similarity.max(), hsi_b_similarity.min())
    # 对hsi_t_tensor的相似度进行排序
    sorted_similarity_t, indices_t = torch.sort(hsi_t_similarity, descending=True)
    # 使用排序后的索引对hsi_t_tensor中的光谱向量进行排序
    sorted_hsi_t_tensor = hsi_t_tensor[indices_t]
    # 对hsi_b_tensor的相似度进行排序
    sorted_similarity_b, indices_b = torch.sort(hsi_b_similarity, descending=True)
    # 使用排序后的索引对hsi_b_tensor中的光谱向量进行排序
    sorted_hsi_b_tensor = hsi_b_tensor[indices_b]

    # AE网络对目标数据进行增广：
    # 先根据已知的目标光谱计算出每个目标光谱和平均光谱之间的平均光谱角相似度, 以这个光谱角相似度作为AE网络恢复出的光谱被接受的阈值
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    dataAugment(hsi_t_tensor, num_augment=len(hsi_b_tensor) * 4, batch_size=16,
                num_epochs=50000, save_dir=save_dir, device=device, type='t',
                num_cluster='all', num_data=len(hsi_list), window_size=window_size)
    dataAugment(hsi_b_tensor, num_augment=len(hsi_b_tensor) * 4, batch_size=64,
                num_epochs=50000, save_dir=save_dir, device=device, type='b',
                num_cluster='all', num_data=len(hsi_list), window_size=window_size)
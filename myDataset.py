
import random
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from utils import maxMinNormalization,\
    cornerPointCalculation, tsGeneration, spectralSimilarity


class CustomDataset(Dataset):
    def __init__(self, hsi_background, hsi_target, ts_tensor, num_clusters=10, algorithm='SA'):
        # 将输入数据存储为实例变量
        self.hsi_background = hsi_background
        self.hsi_target = hsi_target
        self.num_clusters = num_clusters
        self.max_num_per_cluster = 0

        # 使用聚类算法将数据分成10类
        self.background_clusters = self.cluster_data(self.hsi_background, ts_tensor, num_clusters, algorithm)
        self.target_clusters = self.cluster_data(self.hsi_target, ts_tensor, num_clusters, algorithm)

    def cluster_data(self, hsi_tensor, ts_tensor, num_class, algorithm='SA'):
        hsi_similarity = spectralSimilarity(hsi_tensor, ts_tensor, metric='SA')
        print('Maximum similarity between HSI and prior target spectra:', hsi_similarity.max())
        print('Minimum similarity between HSI and prior target spectra:', hsi_similarity.min())
        # 对hsi_t_tensor的相似度进行排序
        sorted_similarity, indices = torch.sort(hsi_similarity, descending=True)
        # 使用排序后的索引对hsi_t_tensor中的光谱向量进行排序
        sorted_hsi_tensor = hsi_tensor[indices]

        hsi_dict = {}
        if algorithm == 'SA':
            num_per_cluster = hsi_tensor.size(0) // num_class
            if num_per_cluster > self.max_num_per_cluster:
                self.max_num_per_cluster = num_per_cluster
            hsi_similarity_dict = {}
            for i in range(num_class):
                if i == num_class - 1:
                    hsi_dict[i] = sorted_hsi_tensor[i * num_per_cluster:]
                    hsi_similarity_dict[i] = sorted_similarity[i * num_per_cluster:]
                else:
                    hsi_dict[i] = sorted_hsi_tensor[i * num_per_cluster:(i + 1) * num_per_cluster]
                    hsi_similarity_dict[i] = sorted_similarity[i * num_per_cluster:(i + 1) * num_per_cluster]
        elif algorithm == "KMeans":
            # 将向量分成 n+2 份
            num_partitions = num_class + 2
            partition_size = len(sorted_hsi_tensor) // num_partitions
            # 选取中间的 n 个向量
            start_idx = partition_size
            end_idx = start_idx + num_class * partition_size
            indices = torch.arange(start_idx, end_idx, partition_size)
            initial_centers = sorted_hsi_tensor[indices].cpu().numpy()
            # 执行 k-means 聚类
            kmeans = KMeans(n_clusters=num_class, init=initial_centers, n_init=1)
            hsi_tensor_np = hsi_tensor.cpu().numpy()
            kmeans.fit(hsi_tensor_np)
            # 创建一个字典来存储每个簇的成员
            for i in range(num_class):
                hsi_dict[i] = torch.from_numpy(hsi_tensor_np[kmeans.labels_ == i])
                if len(hsi_dict[i]) > self.max_num_per_cluster:
                    self.max_num_per_cluster = len(hsi_dict[i])
        return hsi_dict

    def shuffle(self):
        for key in self.background_clusters.keys():
            random.shuffle(self.background_clusters[key])
        for key in self.target_clusters.keys():
            random.shuffle(self.target_clusters[key])

    def __getitem__(self, idx):
        start = idx.start
        batch_size = idx.stop - start
        data = []
        label = []
        # 从每个类别中选取样本
        for key_b in self.background_clusters.keys():
            i = start % len(self.background_clusters[key_b])
            if i + batch_size > len(self.background_clusters[key_b]) - 1:
                data.append(self.background_clusters[key_b][-batch_size:])
            else:
                data.append(self.background_clusters[key_b][i: i + batch_size])
            label += [0] * batch_size
        for key_t in self.target_clusters.keys():
            i = start % len(self.target_clusters[key_t])
            if i + batch_size > len(self.target_clusters[key_t]) - 1:
                data.append(self.target_clusters[key_t][-batch_size:])
            else:
                data.append(self.target_clusters[key_t][i: i + batch_size])
            label += [1] * batch_size
        return torch.cat(data, dim=0), torch.tensor(label)

    def __len__(self):
        # # 返回数量最多的那个簇的样本个数
        # return max(len(self.background_clusters[self.num_clusters - 1]),
        #            len(self.target_clusters[self.num_clusters - 1]), self.max_num_per_cluster)
        return self.max_num_per_cluster

class MyDataLoader():
    def __init__(self, dataset, batch_size=5, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.dataset.shuffle()

    def __iter__(self):
        # 这里可以添加一些在迭代时要做的事情、例如混洗数据等
        for i in range(0, len(self.dataset), self.batch_size):
            # Yield a batch of data
            data_batch, label_batch = self.dataset[i:i+self.batch_size]
            yield data_batch, label_batch

    def __len__(self):
        # 计算有多少个批次, 向上取整
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    MODEL_NAME = 'MSSAE'
    BATCH_SIZE = 128
    type_target_spectrum = 5
    epochs = 200
    in_len = 189
    mapped_len = 128
    d_input = 1
    d_model = 32
    layers = 2
    ffn_hidden = 128
    n_head = 2
    n_layers = 1
    lr = 0.0001
    weight_decay = 1e-5

    """ 数据加载 """
    HSI_DATA_PATH = "dataset/AVIRIS/AVIRIS-I.npy"
    HSI_GT_PATH = "dataset/AVIRIS/AVIRIS-I-gt.npy"
    HSI_DATA_NAME = os.path.splitext(os.path.basename(HSI_DATA_PATH))[0]
    hsi = np.load(HSI_DATA_PATH)
    gt = np.load(HSI_GT_PATH)
    hsi = maxMinNormalization(hsi, dtype=np.float32)
    ts = tsGeneration(hsi, gt, type_target_spectrum)
    # # 角点数据增强
    hsi_w3 = cornerPointCalculation(hsi, window_size=3, metric='Euclidean')
    hsi_w5 = cornerPointCalculation(hsi, window_size=5, metric='Euclidean')
    hsi_w7 = cornerPointCalculation(hsi, window_size=7, metric='Euclidean')
    hsi_list = [hsi, hsi_w3, hsi_w5, hsi_w7]
    hsi_t_list = []
    hsi_b_list = []
    for data in hsi_list:
        hsi_t_list.append(torch.from_numpy(data[np.where(gt == 1)]))
        hsi_b_list.append(torch.from_numpy(data[np.where(gt == 0)]))
    hsi_t_origin_tensor = torch.cat(hsi_t_list, dim=0)
    hsi_t_augment_tensor = torch.from_numpy(np.load(
        'log/myDataAugment/AVIRIS-I/hsi_t_augmented_114840.npy'))
    hsi_t_tensor = torch.cat([hsi_t_origin_tensor, hsi_t_augment_tensor], dim=0)
    label_t_tensor = torch.ones(hsi_t_tensor.shape[0])
    hsi_b_origin_tensor = torch.cat(hsi_b_list, dim=0)
    hsi_b_augment_tensor = torch.from_numpy(np.load(
        'log/myDataAugment/AVIRIS-I/hsi_b_augmented_57360.npy'))
    hsi_b_tensor = torch.cat([hsi_b_origin_tensor, hsi_b_augment_tensor], dim=0)
    ts_tensor = torch.from_numpy(ts)

    dataset = CustomDataset(hsi_b_tensor, hsi_t_tensor, ts_tensor)
    dataloader = MyDataLoader(dataset, batch_size=5, shuffle=True)

    for data, label in dataloader:
        # 随后你可以在这里添加你的模型和损失函数的代码
        pass
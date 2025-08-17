import logging
import os
from time import time
from typing import Union
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# 设置字体, 解决中文乱码问题
from matplotlib import font_manager
font_manager.fontManager.addfont('Fonts/msyh.ttc')
plt.rcParams['font.family'] = 'Microsoft YaHei'

import cv2
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.decomposition import PCA


# 中值滤波
def median_filter(img, kernel_size=3):
    pad_size = kernel_size // 2

    img_pad = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

    img_filter = np.zeros_like(img)
    for i in range(pad_size, img_pad.shape[0] - pad_size):
        for j in range(pad_size, img_pad.shape[1] - pad_size):
            img_filter[i - pad_size, j - pad_size] = np.median(
                img_pad[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1])

    return img_filter

def calculation_Binary_map(detection_map, gt, num_bins=10, folder_display=None):

    detection_map = detection_map.reshape(-1)
    sorted_similarity, indices = torch.sort(torch.from_numpy(detection_map), descending=False)
    # print(sorted_similarity[:20])
    # print(sorted_similarity.max())
    # print(indices[:20])

    # 绘制直方图, 并使用 numpy 的 histogram 函数来获取频率和边界
    hist, bin_edges = np.histogram(detection_map, bins=num_bins)
    my_best_threshold = bin_edges[1]
    Binary_map = np.array([0 if value <= my_best_threshold else 1 for value in detection_map]).reshape(*gt.shape)
    Binary_map_median_filter = median_filter(Binary_map)

    plt.figure()
    plt.imshow(detection_map.reshape(*gt.shape))
    plt.axis("off")
    # plt.title('Binary Map')
    if folder_display is not None:
        os.makedirs(folder_display, exist_ok=True)
        plt.savefig(folder_display + 'detection_map_{}.png'.format(num_bins), bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(Binary_map)
    plt.axis("off")
    # plt.title('Binary Map')
    if folder_display is not None:
        os.makedirs(folder_display, exist_ok=True)
        plt.savefig(folder_display + 'binary_map_{}.png'.format(num_bins), bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(Binary_map_median_filter)
    plt.axis("off")
    # plt.title('Binary Map Median Filter')
    if folder_display is not None:
        os.makedirs(folder_display, exist_ok=True)
        plt.savefig(folder_display + 'binary_map_median_filter_{}.png'.format(num_bins), bbox_inches='tight')
    plt.close()

    return Binary_map.reshape(-1), Binary_map_median_filter.reshape(-1)


def create_logger(log_path, mode='training'):

    """ 创建并配置独立的logger """
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)  # 设置信息级别

    formatter = logging.Formatter('%(asctime)s ==> %(message)s')

    # 创建一个流处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # 创建一个文件处理器, 将日志信息输出到文件中
    log_file_path = os.path.join(log_path, mode + '.log')
    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setFormatter(formatter)

    # 如果 logger 对象没有处理器，则添加处理器
    if not logger.hasHandlers():
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

""" HSI 光谱数据分析"""


def spectralAnalysis(hsi: np.ndarray, gt: np.ndarray, num_samples: int = 10):
    classIndex_list = np.unique(gt)
    for classIndex in classIndex_list:
        pos = [(row, col) for row, col in zip(*np.where(gt == classIndex))]
        plt.figure()
        max_samples = len(pos)
        if max_samples < num_samples:
            num_samples = max_samples
        for i in range(0, num_samples):
            plt.plot(range(hsi.shape[-1]), hsi[pos[i]])
        plt.title('class:{}'.format(classIndex))
        plt.show()
        plt.close()


""" 数据预处理 """


# 最大最小值标准化不会改变光谱曲线的光谱角度【因为 max(hsi) 和 min(hsi)均为常数】
def maxMinNormalization(hsi: Union[np.ndarray, torch.Tensor], dtype=np.float32, mode: str = "Global"):
    # float32 可能会降低准确度
    hsi = np.asarray(hsi, dtype=dtype)

    if np.max(hsi) > 1.0:
        if mode == "Global":
            hsi = (hsi - np.min(hsi)) / (np.max(hsi) - np.min(hsi))
        elif mode == "Local":
            hsi_max = np.expand_dims(np.max(hsi, axis=-1), axis=-1)
            hsi_min = np.expand_dims(np.min(hsi, axis=-1), axis=-1)
            hsi = (hsi - hsi_min) / (hsi_max - hsi_min)
        else:
            print("Unknown Normalization Mode...")
    else:
        print('The maximum value of the data is less than or equal to 1.0,'
              'it seems that standardization is not necessary')

    return hsi

def cornerPointCalculation(data, window_size=3, metric='SA'):
    """

    Args:
        data: HSI
        window_size: option[odd number]
        metric: option['SA', 'Cosine', 'Euclidean']

    Returns:

    """
    print("cornerPointCalculation...")
    start_time = time()
    window_radius = window_size // 2
    # 创建一个 ndarray 用来存放新的数据
    data_new = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
    # padding 操作
    enlar_data = np.zeros((data.shape[0] + window_radius * 2,
                           data.shape[1] + window_radius * 2, data.shape[2]), dtype=np.float32)
    enlar_data[window_radius:data.shape[0] + window_radius, window_radius:data.shape[1] + window_radius, :] = data
    # 将数据做成 tensor
    enlar_data_tensor = torch.from_numpy(enlar_data)
    for x_data in range(data.shape[0]):
        x = x_data + window_radius
        for y_data in range(data.shape[1]):
            y = y_data + window_radius

            cube_data_tensor = enlar_data_tensor[x - window_radius:x + window_radius + 1,
                               y - window_radius:y + window_radius + 1, :].reshape(-1, data.shape[-1])
            cube_center_pixel = enlar_data_tensor[x, y, :].reshape(1, -1)
            weights_vector = spectralSimilarity(cube_data_tensor, cube_center_pixel, metric=metric)
            weights_vector[torch.isnan(weights_vector)] = 0.0
            weights_vector /= weights_vector.sum()
            weighted_sum = torch.mv(cube_data_tensor.t(), weights_vector)

            data_new[x_data, y_data, :] = weighted_sum
    end_time = time()
    print('cornerPointCalculation speed time:', end_time - start_time)
    return np.asarray(data_new)


def cornerPointCalculation_thj(data, window_size=3, metric='SA'):
    """

    Args:
        data: HSI
        window_size: option[odd number]
        metric: option['SA', 'Cosine', 'Euclidean']

    Returns:

    """
    print("cornerPointCalculation...")
    start_time = time()
    window_radius = window_size // 2
    # 创建一个 ndarray 用来存放新的数据
    data_new = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
    # padding 操作
    enlar_data = np.zeros((data.shape[0] + window_radius * 2,
                           data.shape[1] + window_radius * 2, data.shape[2]), dtype=np.float32)
    enlar_data[window_radius:data.shape[0] + window_radius, window_radius:data.shape[1] + window_radius, :] = data
    # 将数据做成 tensor
    enlar_data_tensor = torch.from_numpy(enlar_data)
    for x_data in range(data.shape[0]):
        x = x_data + window_radius
        for y_data in range(data.shape[1]):
            y = y_data + window_radius

            cube_data_tensor = enlar_data_tensor[x - window_radius:x + window_radius + 1,
                               y - window_radius:y + window_radius + 1, :].reshape(-1, data.shape[-1])
            # 中心向量索引
            center_idx = window_radius * (2 * window_radius + 1) + window_radius
            # 去掉中心向量
            mask = torch.ones(cube_data_tensor.shape[0], dtype=torch.bool)
            mask[center_idx] = False
            cube_data_tensor_without_center = cube_data_tensor[mask]

            cube_center_pixel = enlar_data_tensor[x, y, :].reshape(1, -1)
            weights_vector = spectralSimilarity(cube_data_tensor_without_center,
                                                cube_center_pixel, metric=metric)
            weights_vector[torch.isnan(weights_vector)] = 0.0
            weighted_sum = torch.mv(cube_data_tensor_without_center.t(), weights_vector) / len(weights_vector)

            data_new[x_data, y_data, :] = weighted_sum

    end_time = time()
    print('cornerPointCalculation speed time:', end_time - start_time)
    return np.asarray(data_new)

# 对高光谱数据 X 进行 PCA 降维，PCA expected x.dim <= 2
def applyPCA(x, numComponents=150):
    x_shape = x.shape
    x_pca = None
    if len(x_shape) == 3:
        if x_shape[1] == x_shape[2]:
            x = np.reshape(x, (x_shape[0], -1)).transpose(1, 0)
        else:
            x = np.reshape(x, (-1, x_shape[2]))

        print('before PCA x_.shape = ', x.shape)

        random_state = 23
        pca = PCA(n_components=numComponents, whiten=True, random_state=random_state)
        x_pca = pca.fit_transform(x)

        print('after PCA x_pca.shape = ', x_pca.shape)
        x_pca = np.reshape(x_pca, (x_shape[0], x_shape[1], numComponents))
    elif len(x_shape) == 2:
        if x_shape[-1] == 1:
            x = x.transpose(1, 0)
        print('before PCA x_.shape = ', x.shape)

        random_state = 23
        pca = PCA(n_components=numComponents, whiten=True, random_state=random_state)
        x_pca = pca.fit_transform(x)

        print('after PCA x_pca.shape = ', x_pca.shape)

    return x_pca

def tsGeneration(data: np.ndarray, gt: np.ndarray, type: int = 0) -> np.ndarray:
    '''
    using different methods to select the target spectrum ——> 0-2 效果最佳
    :param type:
    0-均值光谱
    1-空间去偏（L2范数）后的均值光谱
    2-光谱去偏（光谱角）后的均值光谱
    3-空间中位光谱（L2范数的中位数）
    4-光谱中位光谱（光谱角的中位数）
    5-距离所有目标像素欧式距离最近的目标像素
    6-距离所有目标像素余弦距离最近的目标像素
    7-距离均值光谱欧式距离最近的目标像素
    8-距离均值光谱余弦距离最近的目标像素
    :return:
    '''

    # 根据目标的位置, 从原始高光谱图像中提取出目标的光谱向量
    ts = data[np.where(gt == np.max(gt))]
    avg_target_spectrum = np.mean(ts, axis=0)
    avg_target_spectrum = np.expand_dims(avg_target_spectrum, axis=0)

    if type == 0:
        return avg_target_spectrum
    elif type == 1:
        spatial_distance = np.sqrt(np.sum((ts - avg_target_spectrum) ** 2, axis=-1))
        arg_distance = np.argsort(spatial_distance)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(removed_deviation_target_spectrum, axis=0)
        return removed_deviation_target_spectrum
    elif type == 2:
        # 计算光谱角 spectral_angle
        dot_product = np.dot(ts, avg_target_spectrum.T)
        # np.linalg.norm 函数来计算向量的范数，默认计算二范数
        norm_product = np.linalg.norm(ts, axis=1) * np.linalg.norm(avg_target_spectrum)
        spectral_cosine = dot_product[:, 0] / norm_product
        # np.clip(num, -1, 1) 将 num 限制在[-1, 1]区间内
        spectral_angle = np.arccos(np.clip(spectral_cosine, -1, 1))
        arg_distance = np.argsort(spectral_angle)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(removed_deviation_target_spectrum, axis=0)
        return removed_deviation_target_spectrum
    elif type == 3:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            dist_list[i] = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=0)
        return mid_target_spectrum
    elif type == 4:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            # 计算光谱角 spectral_angle
            dot_product = np.dot(ts, ts[i].T)
            # np.linalg.norm 函数来计算向量的范数，默认计算二范数
            norm_product = np.linalg.norm(ts, axis=1) * np.linalg.norm(ts[i])
            spectral_cosine = dot_product / norm_product
            # np.clip(num, -1, 1) 将 num 限制在[-1, 1]区间内
            spectral_angle = np.arccos(np.clip(spectral_cosine, -1, 1))
            dist_list[i] = np.mean(spectral_angle)
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=0)
        return mid_target_spectrum
    elif type == 5:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            dist = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=0)
        return target_spectrum
    elif type == 6:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            # 计算光谱角 spectral_angle
            dot_product = np.dot(ts, ts[i].T)
            # np.linalg.norm 函数来计算向量的范数，默认计算二范数
            norm_product = np.linalg.norm(ts, axis=1) * np.linalg.norm(ts[i])
            spectral_cosine = dot_product / norm_product
            # np.clip(num, -1, 1) 将 num 限制在[-1, 1]区间内
            spectral_angle = np.arccos(np.clip(spectral_cosine, -1, 1))
            dist = np.mean(spectral_angle)
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=0)
        return target_spectrum
    elif type == 7:
        distance = np.sqrt(np.sum((ts - avg_target_spectrum) ** 2, axis=-1))
        arg_distance = np.argsort(distance)
        avg_L2_target_spectrum = ts[arg_distance[0]]
        avg_L2_target_spectrum = np.expand_dims(avg_L2_target_spectrum, axis=0)
        return avg_L2_target_spectrum
    elif type == 8:
        # 计算光谱角 spectral_angle
        dot_product = np.dot(ts, avg_target_spectrum.T)
        # np.linalg.norm 函数来计算向量的范数，默认计算二范数
        norm_product = np.linalg.norm(ts, axis=1) * np.linalg.norm(avg_target_spectrum)
        spectral_cosine = dot_product[:, 0] / norm_product
        # np.clip(num, -1, 1) 将 num 限制在[-1, 1]区间内
        spectral_angle = np.arccos(np.clip(spectral_cosine, -1, 1))
        arg_distance = np.argsort(spectral_angle)
        avg_cosin_target_spectrum = ts[arg_distance[0]]
        avg_cosin_target_spectrum = np.expand_dims(avg_cosin_target_spectrum, axis=0)
        return avg_cosin_target_spectrum
    else:
        return avg_target_spectrum


def spectralSimilarity(X_tensor: torch.Tensor, ts_tensor: torch.Tensor,
                       metric: str = 'SA') -> torch.Tensor:
    """
        X: type--tensor
            shape--HW*B
        t: type--tensor
            shape--1*B
    """
    if metric == 'SA':
        # Spectral Angle
        dot_product = torch.mm(X_tensor, ts_tensor.T)
        # torch.norm 函数来计算向量的范数，默认计算二范数
        norm_product = torch.norm(X_tensor, dim=1) * torch.norm(ts_tensor)
        spectral_cosine = dot_product[:, 0] / norm_product
        # torch.clamp(num, -1, 1) 将 num 限制在[-1, 1]区间内
        spectral_cosine = torch.clamp(spectral_cosine, -0.999999, 0.999999)
        spectral_angle = torch.acos(spectral_cosine)
        """
        对于高光谱图像, 每个像素的光谱值通常表示某一特定波长下的光强度或反射率, 全都是非负数
        因此, 实际上不存在“负的”光谱值或"反向”的光谱, 因为光谱值都是非负的, 所以光谱角度的范围是 [0, π/2], 光谱角越小表示相似度越高
        """
        similarity = 1 - spectral_angle / torch.pi
    elif metric == 'Cosine':
        # Cosine Similarity
        dot_product = torch.mm(X_tensor, ts_tensor.T)
        # torch.norm 函数来计算向量的范数，默认计算二范数
        norm_product = torch.norm(X_tensor, dim=1) * torch.norm(ts_tensor)
        spectral_cosine = dot_product[:, 0] / norm_product
        # torch.clamp(num, -1, 1) 将 num 限制在[-1, 1]区间内
        spectral_cosine = torch.clamp(spectral_cosine, -0.999999, 0.999999)
        similarity = (spectral_cosine + 1) / 2  # 将余弦相似度的结果范围从-1~1转换为0~1
    elif metric == 'Euclidean':
        a = 1
        # Euclidean Distance
        distance = torch.sqrt(torch.sum((X_tensor - ts_tensor) ** 2, dim=1))
        similarity = 1 / (a + distance)
    else:
        raise ValueError(f"Unknown metric '{metric}'")
    # 灰度图像中 0表示白色 and 1表示黑色
    return similarity


""" visualization """


def experimentalResultsDisplay(hsi: np.ndarray, gt: np.ndarray, detection_map: np.ndarray,
                               folder_display: str = None, dataset_name: str = 'HSI'):
    if folder_display is not None:
        os.makedirs(folder_display, exist_ok=True)

    # 绘制伪彩色图
    plt.figure()
    rgb_image = hsi.reshape(*gt.shape, -1)[:, :, ::10][:, :, :3]
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title('Pseudo color image')
    if folder_display is not None:
        plt.savefig(folder_display + '0-Pseudo_color_image.png', bbox_inches='tight')
    plt.close()
    # 绘制真实标签图
    plt.figure()
    plt.imshow(gt)
    plt.axis("off")
    plt.title('Ground Truth')
    if folder_display is not None:
        plt.savefig(folder_display + '1-ground_truth.png', bbox_inches='tight')
    plt.close()

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
    # 计算 Youden's J statistic —— Youden index (J)
    J = tpr - fpr
    # 找到具有最大 Youden's J statistic 值的索引
    best_threshold_index = np.argmax(J)
    # 获取最佳阈值
    best_threshold = thresholds[best_threshold_index]
    print('auc_ft={:.{precision}f}, auc_f={:.{precision}f}, auc_t={:.{precision}f}'.format(
        auc_ft, auc_f, auc_t, precision=5))
    print("Best threshold:", best_threshold)

    # 绘制检测图
    plt.figure()
    plt.imshow(detection_map)
    plt.axis("off")
    plt.title('Ground Prediction (AUC = {})'.format(auc_ft))
    if folder_display is not None:
        plt.savefig(folder_display + '{}_ground_prediction.png'.format(dataset_name), bbox_inches='tight')
    plt.show()
    plt.close()
    Binary_map = np.array([1 if value >= best_threshold else 0 for value in y_p]).reshape(*gt.shape)
    plt.figure()
    plt.imshow(Binary_map)
    plt.axis("off")
    plt.title('Binary Map (Threshold = {})'.format(best_threshold))
    if folder_display is not None:
        plt.savefig(folder_display + '{}_binary_map.png'.format(dataset_name), bbox_inches='tight')
    plt.show()
    plt.close()

    # 绘制 2-D ROC曲线
    plt.figure()  # 创建新的图像
    plt.plot(fpr, tpr, label='ROC curve (AUC = %s)' % auc_ft)
    plt.plot([10 ** -4, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xscale("log")  # Set x scale to logarithmic
    plt.xlim([10 ** -4, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if folder_display is not None:
        plt.savefig(folder_display + '{}_2D-ROC-Curve.png'.format(dataset_name), bbox_inches='tight')
    plt.show()
    plt.close()

    # 绘制 3-D ROC曲线
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 改变视角, elev为上下旋转角度, azim为左右旋转角度
    ax.view_init(elev=10, azim=20)  # 使用这个语句来移动 z 轴 ax.zaxis._axinfo['juggled'] = (1, 2, 0) / ax.zaxis.labelpad = 15
    ax.plot3D(thresholds, fpr, tpr, label='3D-ROC curve')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('False Positive Rate')  # 反转Y轴 ax.invert_yaxis()
    # 设置Z轴的标题, 并返回Z轴对象
    zlabel = ax.set_zlabel('True Positive Rate')
    # 获取Z轴的标题，并设置旋转角度
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    zlabel.set_rotation(90)
    plt.title('3D ROC Curve')
    plt.legend(loc="center right")
    if folder_display is not None:
        plt.savefig(folder_display + '{}_3D-ROC-Curve.png'.format(dataset_name), bbox_inches='tight')
    plt.show()
    plt.close()

    # 箱线图分析
    background_pixel_intensities = list(y_p.reshape(-1)[np.where(y_l.reshape(-1) != y_l.max())])
    target_pixel_intensities = list(y_p.reshape(-1)[np.where(y_l.reshape(-1) == y_l.max())])
    # # 将数据保存为MAT文件
    # sio.savemat(folder_display + '{}_background_pixel_intensities.mat'.format(modelConfig["dataset"]), {'data': background_pixel_intensities})
    # sio.savemat(folder_display + '{}_target_pixel_intensities.mat'.format(modelConfig["dataset"]), {'data': target_pixel_intensities})
    data_to_plot = [background_pixel_intensities, target_pixel_intensities]
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    # 创建箱型图  whis--指定上下须和箱体四分位距的比值, 默认为1.5 // widths--标量或者序列，设置箱形图的宽度，默认为0.5
    bp = ax.boxplot(data_to_plot, patch_artist=True, whis=float('inf'), widths=0.05)
    colors = ['#7570b3', '#d95f02']  # 为每种类型选择颜色
    box_plot_type = ['BackGround', 'Target']
    patches = []
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patches.append(mpatches.Patch(color=color))
    # 修改图中其他元素的颜色
    for median in bp['medians']:
        median.set(color='#e7298a', linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=1, linestyle='--')
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=1)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ax.set_xticklabels(box_plot_type)
    # 添加图例
    plt.legend(patches, box_plot_type, loc='upper right')
    # 设置y轴标签
    ax.set_ylabel('Pixel Intensity')
    if folder_display is not None:
        plt.savefig(folder_display + '{}_Box-plot.png'.format(dataset_name), bbox_inches='tight')
    plt.show()
    plt.close()

    return auc_ft, auc_t, auc_f, auc_oa, auc_snpr


def experimentalResultsContrast(hsi: np.ndarray, gt: np.ndarray, detection_maps: list,
                        model_names: list, logger, folder_display: str = None, use_title=True):
    if folder_display is not None:
        os.makedirs(folder_display, exist_ok=True)

    # 绘制伪彩色图
    plt.figure()
    rgb_image = hsi[:, :, ::10][:, :, :3]
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    plt.imshow(rgb_image)
    plt.axis("off")
    if use_title:
        plt.title('Pseudo color image')
    if folder_display is not None:
        plt.savefig(folder_display + '0-Pseudo_color_image.png', bbox_inches='tight')
    plt.close()
    # 绘制真实标签图
    plt.figure()
    plt.imshow(gt)
    plt.axis("off")
    if use_title:
        plt.title('Ground Truth')
    if folder_display is not None:
        plt.savefig(folder_display + '1-ground_truth.png', bbox_inches='tight')
    plt.close()

    # 定义数据变量
    y_target_list = []
    y_background_list = []
    fpr_list = []
    tpr_list = []
    thresholds_list = []
    auc_ft_list = []
    auc_f_list = []
    auc_t_list = []
    best_threshold_list = []
    for model_name, detection_map in zip(model_names, detection_maps):
        logger.info("=" * 20)  # 输出一个80字符长的分隔线
        logger.info(">>Model:{}".format(model_name))
        y_l = gt.reshape(-1)  # 'F' 代表 Fortran 风格 (列优先, 或者称为按列主序)
        y_p = detection_map.reshape(-1)
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
        logger.info('auc_ft: {:.{precision}f}'.format(auc_ft, precision=5))
        logger.info('auc_t: {:.{precision}f}'.format(auc_t, precision=5))
        logger.info('auc_f: {:.{precision}f}'.format(auc_f, precision=5))
        logger.info('auc_oa: {:.{precision}f}'.format(auc_oa, precision=5))
        logger.info('auc_snpr: {:.{precision}f}'.format(auc_snpr, precision=5))
        # 计算 Youden's J statistic —— Youden index (J)
        J = tpr - fpr
        # 找到具有最大 Youden's J statistic 值的索引
        best_threshold_index = np.argmax(J)
        # 获取最佳阈值
        best_threshold = thresholds[best_threshold_index]
        print("Best threshold:", best_threshold)
        logger.info('Best threshold: {:.{precision}f}'.format(best_threshold, precision=5))

        # 绘制检测图
        plt.figure()
        plt.imshow(detection_map)
        plt.axis("off")
        if use_title:
            plt.title('{} Ground Prediction (AUC = {})'.format(model_name, auc_ft))
        if folder_display is not None:
            plt.savefig(folder_display + '{}_ground_prediction.png'.format(model_name), bbox_inches='tight')
        plt.close()

        # 检测图的分布图
        ##########################################################################
        num_bins = 10  # 可以调整此值来改变区间的宽度
        # 绘制直方图
        # 使用 numpy 的 histogram 函数来获取频率和边界
        hist, bin_edges = np.histogram(detection_map, bins=num_bins)
        my_best_threshold = bin_edges[1]
        # print(hist)
        # print(bin_edges)
        plt.hist(detection_map, bins=num_bins, edgecolor='black', alpha=0.7)
        print(detection_map.max())
        print(detection_map.min())
        # 设置图表标题和坐标轴标签
        plt.title('Distribution of Data Points between 0 and 1')
        plt.xlabel('Value Range')
        plt.ylabel('Frequency')
        if use_title:
            plt.title('Distribution of Data Points between 0 and 1')
        if folder_display is not None:
            plt.savefig(folder_display + '{}_ground_prediction_Distribution.png'.format(model_name), bbox_inches='tight')
        # 显示图表
        plt.close()
        ##########################################################################
        # x_indices = np.arange(len(y_p))
        # # 绘制散点图
        # plt.scatter(x_indices, y_p, alpha=0.5, s=10)  # alpha 控制透明度，s 控制点的大小
        ##########################################################################
        # # 设置直方图的区间数量
        # num_bins = 100  # 可以调整此值来改变区间的宽度
        # # 绘制直方图
        # hist, bin_edges = np.histogram(detection_map, bins=num_bins)
        # # 找到直方图的峰值
        # from scipy.signal import find_peaks
        # peaks, _ = find_peaks(hist, height=0)
        # # 从右向左查找峰值之间的谷底
        # optimal_threshold = 0
        # for i in range(len(peaks) - 1, 0, -1):
        #     start = peaks[i - 1]
        #     end = peaks[i]
        #     valley_index = np.argmin(hist[start:end])
        #     current_valley = bin_edges[start + valley_index]
        #     if current_valley > optimal_threshold:
        #         optimal_threshold = current_valley
        # # 绘制直方图和峰值
        # plt.figure(figsize=(10, 6))
        # plt.hist(detection_map, bins=bin_edges, edgecolor='black', alpha=0.7)
        # plt.plot(bin_edges[:-1], hist, color='red')  # 绘制直方图的线图
        # plt.plot(bin_edges[:-1][peaks], hist[peaks], "x", color='blue')  # 标记峰值
        # # 标记分界点
        # plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Threshold: {optimal_threshold:.4f}')
        # # 设置图表标题和坐标轴标签
        # plt.xlabel('Value Range')
        # plt.ylabel('Frequency')
        # # 根据条件设置图表标题
        # if use_title:
        #     plt.title('Distribution of Data Points between 0 and 1')
        # # 根据条件保存图表
        # if folder_display is not None:
        #     plt.savefig(folder_display + '{}_ground_prediction_Distribution.png'.format(model_name),
        #                 bbox_inches='tight')
        # plt.close()
        # print("Optimal threshold:", optimal_threshold)
        ##########################################################################
        # 设置直方图的区间数量
        # num_bins = 100  # 可以调整此值来改变区间的宽度
        # # 绘制直方图
        # hist, bin_edges = np.histogram(detection_map, bins=num_bins)
        # # 找到直方图的峰值
        # from scipy.signal import find_peaks
        # peaks, _ = find_peaks(hist, height=0)
        # # 查找峰值之间的谷底
        # valleys = []
        # for i in range(len(peaks) - 1):
        #     start = peaks[i]
        #     end = peaks[i + 1]
        #     valley_index = np.argmin(hist[start:end])
        #     valleys.append(bin_edges[start + valley_index])
        # # 找到最深的谷底作为分界点
        # optimal_threshold = min(valleys)
        # # 绘制直方图和峰值
        # plt.figure(figsize=(10, 6))
        # plt.hist(detection_map, bins=bin_edges, edgecolor='black', alpha=0.7)
        # plt.plot(bin_edges[:-1], hist, color='red')  # 绘制直方图的线图
        # plt.plot(bin_edges[:-1][peaks], hist[peaks], "x", color='blue')  # 标记峰值
        # # 标记分界点
        # plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Threshold: {optimal_threshold:.4f}')
        # # 设置图表标题和坐标轴标签
        # plt.xlabel('Value Range')
        # plt.ylabel('Frequency')
        # # 根据条件设置图表标题
        # if use_title:
        #     plt.title('Distribution of Data Points between 0 and 1')
        # # 根据条件保存图表
        # if folder_display is not None:
        #     plt.savefig(folder_display + '{}_ground_prediction_Distribution.png'.format(model_name),
        #                 bbox_inches='tight')
        # plt.close()
        # print("Optimal threshold:", optimal_threshold)

        # 绘制检测二值图
        # 对hsi_t_tensor的相似度进行排序
        sorted_similarity, indices = torch.sort(torch.from_numpy(y_p.reshape(-1)), descending=False)
        # print(sorted_similarity[:20])
        # print(sorted_similarity.max())
        # print(indices[:20])
        # Binary_map = np.array([0 if value <= sorted_similarity[-len(y_p) // 100] else 1 for value in y_p]).reshape(*gt.shape)
        Binary_map = np.array([0 if value <= bin_edges[1] else 1 for value in y_p]).reshape(*gt.shape)
        # Binary_map = np.array([0 if value <= best_threshold else 1 for value in y_p]).reshape(*gt.shape)
        plt.figure()
        plt.imshow(Binary_map)
        plt.axis("off")
        if use_title:
            plt.title('{} Binary Map (Threshold = {})'.format(model_name, best_threshold))
        if folder_display is not None:
            plt.savefig(folder_display + '{}_binary_map.png'.format(model_name), bbox_inches='tight')
        plt.close()
        # 绘制检测中值滤波二值图
        Binary_map_median_filter = median_filter(Binary_map)
        plt.figure()
        plt.imshow(Binary_map_median_filter)
        plt.axis("off")
        if use_title:
            plt.title('{} Binary Map Median Filter)'.format(model_name))
        if folder_display is not None:
            plt.savefig(folder_display + '{}_binary_map_median_filter.png'.format(model_name), bbox_inches='tight')
        plt.close()

        # 添加数据
        y_background_list.append(list(y_p.reshape(-1)[np.where(y_l.reshape(-1) != y_l.max())]))
        y_target_list.append(list(y_p.reshape(-1)[np.where(y_l.reshape(-1) == y_l.max())]))
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        thresholds_list.append(thresholds)
        auc_ft_list.append(auc_ft)
        auc_f_list.append(auc_f)
        auc_t_list.append(auc_t)
        best_threshold_list.append(best_threshold)

    # 定义颜色列表
    # colors = ['#0066cc', '#ff4500', '#00ff00', '#ffff00', '#00bfff',
    #           '#ff69b4', '#800000', '#008000', '#000080', '#808000']
    colors = ['#0066cc', '#ff4500', '#00FFFF', '#ffff00',
              '#ff00ff', '#0a0404', '#800000', '#00ff7f', '#808000']
    # from matplotlib.font_manager import FontProperties
    # # 设置图例的字体属性
    # fontP = FontProperties()
    # fontP.set_size('large')  # 可以调整大小
    # fontP.set_weight('bold')  # 设置字体加粗


    # 绘制 2-D-FT ROC曲线
    plt.figure(figsize=(12, 8))  # 创建新的图像
    # for i, model_name in enumerate(model_names):
    #     plt.plot(fpr_list[i], tpr_list[i], color=colors[i],
    #              label='%s (AUC = %s)' % (model_name, auc_ft_list[i]))
    for i, model_name in enumerate(model_names):
        plt.plot(fpr_list[i], tpr_list[i], color=colors[i], label='%s' % model_name, linewidth=3)
    # plt.plot([10**-4, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xscale("log")  # Set x scale to logarithmic
    plt.xlim([10 ** -4, 1])
    plt.ylim([0.0, 1.05])
    # 增大刻度数字的大小
    plt.tick_params(axis='both', labelsize=20)  # 调整主要刻度标签的大小
    plt.xlabel('False Alarm Rate', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=30)
    if use_title:
        plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=25)
    plt.grid(True, alpha=0.3)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 2D-FT-ROC-Curve.png', bbox_inches='tight')
    plt.close()
    # 绘制 2-D-F ROC曲线
    plt.figure(figsize=(12, 8))  # 创建新的图像
    # for i, model_name in enumerate(model_names):
    #     plt.plot(thresholds_list[i], fpr_list[i], color=colors[i],
    #              label='%s (AUC = %s)' % (model_name, auc_f_list[i]))
    for i, model_name in enumerate(model_names):
        plt.plot(thresholds_list[i], fpr_list[i], color=colors[i], label='%s' % model_name, linewidth=3)
    # plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xscale("log")  # Set x scale to logarithmic
    plt.xlim([10 ** -2, 1])
    plt.xlim([0, 1])
    plt.ylim([0.0, 1.05])
    # 增大刻度数字的大小
    plt.tick_params(axis='both', labelsize=20)  # 调整主要刻度标签的大小
    plt.xlabel(r'$\tau$', fontsize=30)
    plt.ylabel('False Alarm Rate', fontsize=30)
    if use_title:
        plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=25)
    plt.grid(True, alpha=0.3)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 2D-F-ROC-Curve.png', bbox_inches='tight')
    plt.close()
    # 绘制 2-D-T ROC曲线
    plt.figure(figsize=(12, 8))  # 创建新的图像
    # for i, model_name in enumerate(model_names):
    #     plt.plot(thresholds_list[i], tpr_list[i], color=colors[i],
    #              label='%s (AUC = %s)' % (model_name, auc_t_list[i]))
    for i, model_name in enumerate(model_names):
        plt.plot(thresholds_list[i], tpr_list[i], color=colors[i], label='%s' % model_name, linewidth=3)
    # plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0, 1])
    plt.ylim([0.0, 1.05])
    # 增大刻度数字的大小
    plt.tick_params(axis='both', labelsize=20)  # 调整主要刻度标签的大小
    plt.xlabel(r'$\tau$', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=30)
    if use_title:
        plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=25)
    plt.grid(True, alpha=0.3)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 2D-T-ROC-Curve.png', bbox_inches='tight')
    plt.close()

    # 绘制 3-D ROC 曲线
    fig_3D_plot = plt.figure()
    ax_3D_plot = fig_3D_plot.add_subplot(111, projection='3d')
    # 改变视角, elev为上下旋转角度, azim为左右旋转角度
    ax_3D_plot.view_init(elev=10, azim=45)
    for i, model_name in enumerate(model_names):
        # 在 plot3D 函数中添加 color 参数来设置颜色
        ax_3D_plot.plot3D(thresholds_list[i], fpr_list[i], tpr_list[i],
                          color=colors[i], label='%s' % model_name)
    ax_3D_plot.set_xlabel(r'$\tau$', fontsize=12)
    ax_3D_plot.set_ylabel('False Alarm Rate', fontsize=12)
    zlabel = ax_3D_plot.set_zlabel('True Positive Rate', fontsize=12)
    ax_3D_plot.zaxis.set_rotate_label(False)  # disable automatic rotation
    zlabel.set_rotation(90)
    if use_title:
        plt.title('3D ROC Curve')
    plt.legend(loc="center right", fontsize=12)
    ax_3D_plot.grid(True, color='gray', alpha=0.1)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 3D-ROC-Curve.png')
    plt.close()

    """ 箱线图对比分析 """
    fig_box_plot = plt.figure(figsize=(8, 6))
    ax_box_plot = fig_box_plot.add_subplot(facecolor='white')
    # 定义数据
    boxplot_data = [y_background_list, y_target_list]
    x_labels = model_names
    legend_labels = ['BackGround', 'Target']
    color_list = ['#FF00CC', '#0c35eb']
    group_data_number = len(legend_labels)
    length = len(x_labels)
    x_loc = np.arange(length)
    # 如何绘制多组箱线图
    total_width = 0.4  # 一个组所有箱线图总的宽度, 包含了箱体和箱体间距, 取值为0.4
    box_total_width = total_width * 0.8  # 一个组所有箱体总的宽度
    interval_total_width = total_width * 0.2  # 一个组所有箱体间距总的宽度
    box_width = box_total_width / group_data_number
    ###################### 计算两两箱体间距 #############################
    if group_data_number == 1:
        interval_width = interval_total_width
    else:
        interval_width = interval_total_width / (group_data_number - 1)
    ###################### 计算每个箱体的x轴坐标 #############################
    if group_data_number % 2 == 0:
        x1_box = x_loc - (group_data_number / 2 - 1) * box_width - box_width / 2 - \
                 (group_data_number / 2 - 1) * interval_width - interval_width / 2
    else:
        x1_box = x_loc - ((group_data_number - 1) / 2) * box_width - ((group_data_number - 1) / 2) * interval_width
    x_list_box = [x1_box + box_width * i + interval_width * i for i in range(group_data_number)]
    patches = []  # 初始化空列表在循环外部
    for i in range(len(boxplot_data)):
        color = color_list[i]
        # boxplot_data_num用来统计每组数据的长度, 画scatter图时会用到
        boxplot_data_num = []
        for j in boxplot_data[i]:
            boxplot_data_num_tmp = len(j)
            boxplot_data_num.append(boxplot_data_num_tmp)
        #######################
        bp = ax_box_plot.boxplot(boxplot_data[i], positions=x_list_box[i], widths=box_width,
                                 patch_artist=True, whis=float('inf'),
                                 medianprops={'lw': 1, 'color': 'black'},
                                 boxprops={'lw': 1, 'facecolor': 'None', 'edgecolor': color},
                                 capprops={'lw': 1, 'color': color},
                                 whiskerprops={'ls': '--', 'lw': 1, 'color': color},
                                 showfliers=False, zorder=1)
        patches.append(mpatches.Patch(color=color))
    ax_box_plot.grid(True, ls=':', color='b', alpha=0.3)
    if use_title:
        plt.title('不同算法背景/目标分离度分析', fontweight='bold')
    ax_box_plot.set_xticks(x_loc)
    ax_box_plot.set_xticklabels(x_labels, rotation=45)
    ax_box_plot.set_ylabel('normalized statistic range', fontweight='bold', fontsize=20)
    ################################################################################################################
    # bbox_to_anchor 参数将图例相对于包含它的轴的位置进行定位
    # plt.legend(patches, legend_labels, loc='upper right', bbox_to_anchor=(1, 1),
    #            facecolor='None', edgecolor='#000000', frameon=True, ncol=1, markerscale=3, borderaxespad=0,
    #            handletextpad=0.1, fontsize='x-large', title_fontsize='x-large', handlelength=2, handleheight=0)
    plt.legend(patches, legend_labels, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 20},
               facecolor='None', edgecolor='#000000', frameon=True, ncol=1, markerscale=3, borderaxespad=0,
               handletextpad=0.1, fontsize='x-large', title_fontsize='x-large', handlelength=2, handleheight=0)
    ################################################################################################################
    plt.xticks(weight='bold', fontsize=20)
    plt.yticks(weight='bold', fontsize=15)
    fig_box_plot.tight_layout()
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast Box-plot.png', bbox_inches='tight')
    plt.close()




def ablationExperimentalResultsContrast(hsi: np.ndarray, gt: np.ndarray, detection_maps: list,
                                model_names: list, logger, folder_display: str = None, use_title=True):
    if folder_display is not None:
        os.makedirs(folder_display, exist_ok=True)

    # 绘制伪彩色图
    plt.figure()
    rgb_image = hsi[:, :, ::10][:, :, :3]
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    plt.imshow(rgb_image)
    plt.axis("off")
    if use_title:
        plt.title('Pseudo color image')
    if folder_display is not None:
        plt.savefig(folder_display + '0-Pseudo_color_image.png', bbox_inches='tight')
    plt.close()
    # 绘制真实标签图
    plt.figure()
    plt.imshow(gt)
    plt.axis("off")
    if use_title:
        plt.title('Ground Truth')
    if folder_display is not None:
        plt.savefig(folder_display + '1-ground_truth.png', bbox_inches='tight')
    plt.close()

    # 定义数据变量
    y_target_list = []
    y_background_list = []
    fpr_list = []
    tpr_list = []
    thresholds_list = []
    auc_ft_list = []
    auc_f_list = []
    auc_t_list = []
    best_threshold_list = []
    for model_name, detection_map in zip(model_names, detection_maps):
        logger.info("=" * 20)  # 输出一个80字符长的分隔线
        logger.info(">>Model:{}".format(model_name))
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
        logger.info('auc_ft: {:.{precision}f}'.format(auc_ft, precision=5))
        logger.info('auc_t: {:.{precision}f}'.format(auc_t, precision=5))
        logger.info('auc_f: {:.{precision}f}'.format(auc_f, precision=5))
        logger.info('auc_oa: {:.{precision}f}'.format(auc_oa, precision=5))
        logger.info('auc_snpr: {:.{precision}f}'.format(auc_snpr, precision=5))
        # 计算 Youden's J statistic —— Youden index (J)
        J = tpr - fpr
        # 找到具有最大 Youden's J statistic 值的索引
        best_threshold_index = np.argmax(J)
        # 获取最佳阈值
        best_threshold = thresholds[best_threshold_index]
        print("Best threshold:", best_threshold)

        # 绘制检测图
        plt.figure()
        plt.imshow(detection_map)
        plt.axis("off")
        if use_title:
            plt.title('{} Ground Prediction (AUC = {})'.format(model_name, auc_ft))
        if folder_display is not None:
            plt.savefig(folder_display + '{}_ground_prediction.png'.format(model_name), bbox_inches='tight')
        plt.close()
        Binary_map = np.array([1 if value >= best_threshold else 0 for value in y_p]).reshape(*gt.shape)
        plt.figure()
        plt.imshow(Binary_map)
        plt.axis("off")
        if use_title:
            plt.title('{} Binary Map (Threshold = {})'.format(model_name, best_threshold))
        if folder_display is not None:
            plt.savefig(folder_display + '{}_binary_map.png'.format(model_name), bbox_inches='tight')
        plt.close()

        # 添加数据
        y_background_list.append(list(y_p.reshape(-1)[np.where(y_l.reshape(-1) != y_l.max())]))
        y_target_list.append(list(y_p.reshape(-1)[np.where(y_l.reshape(-1) == y_l.max())]))
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        thresholds_list.append(thresholds)
        auc_ft_list.append(auc_ft)
        auc_f_list.append(auc_f)
        auc_t_list.append(auc_t)
        best_threshold_list.append(best_threshold)

    # 定义颜色列表
    # colors = ['#0066cc', '#ff4500', '#00ff00', '#ffff00', '#00bfff',
    #           '#ff69b4', '#800000', '#008000', '#000080', '#808000']
    colors = ['#0066cc', '#ff4500', '#00FFFF', '#ffff00',
              '#ff00ff', '#0a0404', '#800000', '#00ff7f']
    # from matplotlib.font_manager import FontProperties
    # # 设置图例的字体属性
    # fontP = FontProperties()
    # fontP.set_size('large')  # 可以调整大小
    # fontP.set_weight('bold')  # 设置字体加粗


    # 绘制 2-D-FT ROC曲线
    plt.figure(figsize=(12, 8))  # 创建新的图像
    # for i, model_name in enumerate(model_names):
    #     plt.plot(fpr_list[i], tpr_list[i], color=colors[i],
    #              label='%s (AUC = %s)' % (model_name, auc_ft_list[i]))
    for i, model_name in enumerate(model_names):
        plt.plot(fpr_list[i], tpr_list[i], color=colors[i], label='%s' % model_name, linewidth=3)
        # if model_name != 'BTSNet':
        #     plt.plot(fpr_list[i], tpr_list[i], color=colors[i], label='w/o %s' % model_name, linewidth=3)
        # else:
        #     plt.plot(fpr_list[i], tpr_list[i], color=colors[i], label='%s' % model_name, linewidth=3)
    # plt.plot([10**-4, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xscale("log")  # Set x scale to logarithmic
    plt.xlim([10 ** -4, 1])
    plt.ylim([0.0, 1.05])
    # 增大刻度数字的大小
    plt.tick_params(axis='both', labelsize=20)  # 调整主要刻度标签的大小
    plt.xlabel('False Alarm Rate', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=30)
    if use_title:
        plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=20)
    plt.grid(True, alpha=0.3)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 2D-FT-ROC-Curve.png', bbox_inches='tight')
    plt.close()

    # 绘制 2-D-F ROC曲线
    plt.figure(figsize=(12, 8))  # 创建新的图像
    for i, model_name in enumerate(model_names):
        plt.plot(thresholds_list[i], fpr_list[i], color=colors[i],
                 label='%s' % model_name, linewidth=3)
        # if model_name != 'BTSNet':
        #     plt.plot(thresholds_list[i], fpr_list[i], color=colors[i],
        #              label='w/o %s' % model_name, linewidth=3)
        # else:
        #     plt.plot(thresholds_list[i], fpr_list[i], color=colors[i],
        #              label='%s' % model_name, linewidth=3)
    # plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xscale("log")  # Set x scale to logarithmic
    plt.xlim([10 ** -3, 1])
    plt.ylim([0.0, 1.05])
    # 增大刻度数字的大小
    plt.tick_params(axis='both', labelsize=20)  # 调整主要刻度标签的大小
    plt.xlabel(r'$\tau$', fontsize=30)
    plt.ylabel('False Alarm Rate', fontsize=30)
    if use_title:
        plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=20)
    plt.grid(True, alpha=0.3)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 2D-F-ROC-Curve.png', bbox_inches='tight')
    plt.close()

    # 绘制 2-D-T ROC曲线
    plt.figure(figsize=(12, 8))  # 创建新的图像
    # for i, model_name in enumerate(model_names):
    #     plt.plot(thresholds_list[i], tpr_list[i], color=colors[i],
    #              label='%s (AUC = %s)' % (model_name, auc_t_list[i]))
    for i, model_name in enumerate(model_names):
        plt.plot(thresholds_list[i], tpr_list[i], color=colors[i],
                 label='%s' % model_name, linewidth=3)
        # if model_name != 'BTSNet':
        #     plt.plot(thresholds_list[i], tpr_list[i], color=colors[i],
        #              label='w/o %s' % model_name, linewidth=3)
        # else:
        #     plt.plot(thresholds_list[i], tpr_list[i], color=colors[i],
        #              label='%s' % model_name, linewidth=3)
    # plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xscale("log")  # Set x scale to logarithmic
    plt.xlim([10 ** -3, 1])
    plt.ylim([0.0, 1.05])
    # 增大刻度数字的大小
    plt.tick_params(axis='both', labelsize=20)  # 调整主要刻度标签的大小
    plt.xlabel(r'$\tau$', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=30)
    if use_title:
        plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=20)
    plt.grid(True, alpha=0.3)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 2D-T-ROC-Curve.png', bbox_inches='tight')
    plt.close()

    # 绘制 3-D ROC 曲线
    fig_3D_plot = plt.figure()
    ax_3D_plot = fig_3D_plot.add_subplot(111, projection='3d')
    # 改变视角, elev为上下旋转角度, azim为左右旋转角度
    ax_3D_plot.view_init(elev=10, azim=45)
    for i, model_name in enumerate(model_names):
        ax_3D_plot.plot3D(thresholds_list[i], fpr_list[i], tpr_list[i],
                          color=colors[i], label='%s' % model_name)
        # if model_name != 'BTSNet':
        #     # 在 plot3D 函数中添加 color 参数来设置颜色
        #     ax_3D_plot.plot3D(thresholds_list[i], fpr_list[i], tpr_list[i],
        #                       color=colors[i], label='w/o %s' % model_name)
        # else:
        #     # 在 plot3D 函数中添加 color 参数来设置颜色
        #     ax_3D_plot.plot3D(thresholds_list[i], fpr_list[i], tpr_list[i],
        #                       color=colors[i], label='%s' % model_name)
    ax_3D_plot.set_xlabel(r'$\tau$', fontsize=12)
    ax_3D_plot.set_ylabel('False Alarm Rate', fontsize=12)
    zlabel = ax_3D_plot.set_zlabel('True Positive Rate', fontsize=12)
    ax_3D_plot.zaxis.set_rotate_label(False)  # disable automatic rotation
    zlabel.set_rotation(90)
    if use_title:
        plt.title('3D ROC Curve')
    plt.legend(loc="center right", fontsize=8)
    ax_3D_plot.grid(True, color='gray', alpha=0.1)
    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast 3D-ROC-Curve.png')
    plt.close()


    """ 箱线图对比分析 """
    fig_box_plot = plt.figure(figsize=(8, 6))
    ax_box_plot = fig_box_plot.add_subplot(facecolor='white')
    # 定义数据
    boxplot_data = [y_background_list, y_target_list]
    x_labels = []
    for model_name in model_names:
        x_labels.append(model_name)
        # if model_name != 'BTSNet':
        #     x_labels.append('w/o ' + model_name)
        # else:
        #     x_labels.append('BTSNet')
    legend_labels = ['BackGround', 'Target']
    color_list = ['#FF00CC', '#0c35eb']
    group_data_number = len(legend_labels)
    length = len(x_labels)
    x_loc = np.arange(length)
    # 计算箱体宽度和间距
    total_width = 0.4  # 一个组所有箱线图总的宽度, 包含了箱体和箱体间距, 取值为0.4
    box_total_width = total_width * 0.8  # 一个组所有箱体总的宽度
    interval_total_width = total_width * 0.2  # 一个组所有箱体间距总的宽度
    box_width = box_total_width / group_data_number
    # 计算两两箱体间距
    if group_data_number == 1:
        interval_width = interval_total_width
    else:
        interval_width = interval_total_width / (group_data_number - 1)
    # 计算每个箱体的x轴坐标
    if group_data_number % 2 == 0:
        x1_box = (x_loc - (group_data_number / 2 - 1) * box_width - box_width / 2
                  - (group_data_number / 2 - 1) * interval_width - interval_width / 2)
    else:
        x1_box = x_loc - ((group_data_number - 1) / 2) * box_width - ((group_data_number - 1) / 2) * interval_width
    x_list_box = [x1_box + box_width * i + interval_width * i for i in range(group_data_number)]
    patches = []  # 初始化空列表在循环外部
    for i in range(len(boxplot_data)):
        color = color_list[i]
        bp = ax_box_plot.boxplot(boxplot_data[i], positions=x_list_box[i], widths=box_width,
                                 patch_artist=True, whis=float('inf'),
                                 medianprops={'lw': 1, 'color': 'black'},
                                 boxprops={'lw': 1, 'facecolor': 'None', 'edgecolor': color},
                                 capprops={'lw': 1, 'color': color},
                                 whiskerprops={'ls': '--', 'lw': 1, 'color': color},
                                 showfliers=False, zorder=1)
        patches.append(mpatches.Patch(color=color))
    ax_box_plot.grid(True, ls=':', color='b', alpha=0.3)
    if use_title:
        plt.title('不同算法背景/目标分离度分析', fontweight='bold')
    # 设置x轴标签
    for i, label in enumerate(x_labels):
        ax_box_plot.text(x_list_box[0][i] - box_total_width / 4, 0.5, label, rotation=90, va='center', ha='center', fontsize=15)
    ax_box_plot.set_xticklabels([])  # 移除x轴标签
    ax_box_plot.set_ylabel('normalized statistic range', fontweight='bold', fontsize=15)
    # 添加图例
    plt.legend(patches, legend_labels, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 20},
               facecolor='None', edgecolor='#000000', frameon=True, ncol=1, markerscale=3, borderaxespad=0,
               handletextpad=0.1, fontsize='x-large', title_fontsize='x-large', handlelength=2, handleheight=0)
    plt.xticks(weight='bold', fontsize=15)
    plt.yticks(weight='bold', fontsize=15)
    fig_box_plot.tight_layout()

    if folder_display is not None:
        plt.savefig(folder_display + 'Model-Contrast Box-plot.png', bbox_inches='tight')
    plt.close()


    # fig_box_plot = plt.figure(figsize=(8, 6))
    # ax_box_plot = fig_box_plot.add_subplot(facecolor='white')
    # # 定义数据
    # boxplot_data = [y_background_list, y_target_list]
    # x_labels = []
    # for model_name in model_names:
    #     if model_name != 'MSSAE':
    #         x_labels.append('w/o ' + model_name)
    #     else:
    #         x_labels.append('MSSAE')
    # legend_labels = ['BackGround', 'Target']
    # color_list = ['#FF00CC', '#0c35eb']
    # group_data_number = len(legend_labels)
    # length = len(x_labels)
    # x_loc = np.arange(length)
    # # 如何绘制多组箱线图
    # total_width = 0.4  # 一个组所有箱线图总的宽度, 包含了箱体和箱体间距, 取值为0.4
    # box_total_width = total_width * 0.8  # 一个组所有箱体总的宽度
    # interval_total_width = total_width * 0.2  # 一个组所有箱体间距总的宽度
    # box_width = box_total_width / group_data_number
    # ###################### 计算两两箱体间距 #############################
    # if group_data_number == 1:
    #     interval_width = interval_total_width
    # else:
    #     interval_width = interval_total_width / (group_data_number - 1)
    # ###################### 计算每个箱体的x轴坐标 #############################
    # if group_data_number % 2 == 0:
    #     x1_box = x_loc - (group_data_number / 2 - 1) * box_width - box_width / 2 - \
    #              (group_data_number / 2 - 1) * interval_width - interval_width / 2
    # else:
    #     x1_box = x_loc - ((group_data_number - 1) / 2) * box_width - ((group_data_number - 1) / 2) * interval_width
    # x_list_box = [x1_box + box_width * i + interval_width * i for i in range(group_data_number)]
    # patches = []  # 初始化空列表在循环外部
    # for i in range(len(boxplot_data)):
    #     color = color_list[i]
    #     # boxplot_data_num用来统计每组数据的长度, 画scatter图时会用到
    #     boxplot_data_num = []
    #     for j in boxplot_data[i]:
    #         boxplot_data_num_tmp = len(j)
    #         boxplot_data_num.append(boxplot_data_num_tmp)
    #     #######################
    #     bp = ax_box_plot.boxplot(boxplot_data[i], positions=x_list_box[i], widths=box_width,
    #                              patch_artist=True, whis=float('inf'),
    #                              medianprops={'lw': 1, 'color': 'black'},
    #                              boxprops={'lw': 1, 'facecolor': 'None', 'edgecolor': color},
    #                              capprops={'lw': 1, 'color': color},
    #                              whiskerprops={'ls': '--', 'lw': 1, 'color': color},
    #                              showfliers=False, zorder=1)
    #     patches.append(mpatches.Patch(color=color))
    # ax_box_plot.grid(True, ls=':', color='b', alpha=0.3)
    # if use_title:
    #     plt.title('不同算法背景/目标分离度分析', fontweight='bold')
    # ax_box_plot.set_xticks(x_loc)
    # ax_box_plot.set_xticklabels(x_labels, rotation=30)
    # ax_box_plot.set_ylabel('normalized statistic range', fontweight='bold', fontsize=15)
    # ################################################################################################################
    # # bbox_to_anchor 参数将图例相对于包含它的轴的位置进行定位
    # # plt.legend(patches, legend_labels, loc='upper right', bbox_to_anchor=(1, 1),
    # #            facecolor='None', edgecolor='#000000', frameon=True, ncol=1, markerscale=3, borderaxespad=0,
    # #            handletextpad=0.1, fontsize='x-large', title_fontsize='x-large', handlelength=2, handleheight=0)
    # plt.legend(patches, legend_labels, loc='upper left', bbox_to_anchor=(0, 1.35), prop={'size': 20},
    #            facecolor='None', edgecolor='#000000', frameon=True, ncol=1, markerscale=3, borderaxespad=0,
    #            handletextpad=0.1, fontsize='x-large', title_fontsize='x-large', handlelength=2, handleheight=0)
    # ################################################################################################################
    # plt.xticks(weight='bold', fontsize=15)
    # plt.yticks(weight='bold', fontsize=15)
    # fig_box_plot.tight_layout()
    # if folder_display is not None:
    #     plt.savefig(folder_display + 'Model-Contrast Box-plot.png', bbox_inches='tight')
    # plt.close()
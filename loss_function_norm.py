import math
import torch

from utils import spectralSimilarity


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_encode_loss(encoded_x, encoded_ts, y, similarity_coefficient, alaf=0.2, metric='SA'):
    """

    Args:
        encoded_x:    模型编码器的光谱向量
        y:              每个样本对应的标签
        encoded_ts:   先验目标向量

    Returns:            损失函数

    """
    # print(max(similarity_coefficient))
    # print(min(similarity_coefficient))
    similarity = spectralSimilarity(encoded_x, encoded_ts, metric)
    similarity_coefficient = alaf + (similarity_coefficient - similarity_coefficient.min()
            ) / (similarity_coefficient.max() - similarity_coefficient.min())
    # print(max(similarity_coefficient))
    # print(min(similarity_coefficient))
    pos_t = torch.where(y == 1)
    pos_b = torch.where(y == 0)
    loss_t = ((torch.exp(1 / similarity_coefficient[pos_t])) * (1 - similarity[pos_t])).sum()
    loss_b = ((torch.exp(similarity_coefficient[pos_b])) * similarity[pos_b]).sum()
    # loss_t = ((torch.exp(1 / similarity_coefficient[pos_t]))).sum()
    # loss_b = ((torch.exp(similarity_coefficient[pos_b]))).sum()
    loss = (loss_t + loss_b) / len(y)

    return loss

def calculate_decode_loss(x, recon_x, sa_f=0.9):
    """

    Args:
        x:              原始的光谱向量
        recon_x:        重建的光谱向量

    Returns:            重建损失函数

    """
    dot_product = torch.mm(x, recon_x.T)
    # 获取对角线元素
    diagonal_elements = torch.diag(dot_product)
    # torch.norm 函数来计算向量的范数，默认计算二范数
    norm_product = torch.norm(x, dim=1) * torch.norm(recon_x, dim=1)
    spectral_cosine = diagonal_elements / norm_product
    # torch.clamp(num, -1, 1) 将 num 限制在[-1, 1]区间内
    spectral_cosine = torch.clamp(spectral_cosine, -0.999999, 0.999999)
    spectral_angle = torch.acos(spectral_cosine)
    loss_sa = torch.mean(spectral_angle / torch.pi)

    # 添加 MSE 损失
    mse_loss_fn = torch.nn.MSELoss()
    loss_mse = mse_loss_fn(x, recon_x)

    # 把两个损失组合起来
    loss = sa_f * loss_sa + (1 - sa_f) * loss_mse

    return loss, loss_sa, loss_mse

def calculate_loss(x, encoded_x, encoded_ts, recon_x, y,
            similarity_coefficient, encode_f=0.5, alaf=0.2, metric='SA'):

    encoded_loss = calculate_encode_loss(encoded_x, encoded_ts, y, similarity_coefficient, alaf, metric)
    decoded_loss, decoded_loss_sa, decoded_loss_mse = calculate_decode_loss(x, recon_x)

    ae_loss = encode_f * encoded_loss + (1 - encode_f) * decoded_loss
    # ae_loss = math.exp(10 * encode_f) * encoded_loss + math.exp(10 * (1 - encode_f)) * decoded_loss

    return ae_loss, encoded_loss, decoded_loss


# import math
# import torch
#
# from utils import spectralSimilarity
#
#
# class AverageMeter(object):
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# def calculate_encode_loss(encoded_x, encoded_ts, y, similarity_coefficient, alaf=0.2, metric='SA'):
#     """
#
#     Args:
#         encoded_x:    模型编码器的光谱向量
#         y:              每个样本对应的标签
#         encoded_ts:   先验目标向量
#
#     Returns:            损失函数
#
#     """
#     # print(max(similarity_coefficient))
#     # print(min(similarity_coefficient))
#     similarity = spectralSimilarity(encoded_x, encoded_ts, metric)
#     similarity_coefficient = alaf + (similarity_coefficient - similarity_coefficient.min()
#             ) / (similarity_coefficient.max() - similarity_coefficient.min())
#     # print(max(similarity_coefficient))
#     # print(min(similarity_coefficient))
#     pos_t = torch.where(y == 1)
#     pos_b = torch.where(y == 0)
#     loss_t = ((torch.exp(1 / similarity_coefficient[pos_t])) * (1 - similarity[pos_t])).sum()
#     loss_b = ((torch.exp(similarity_coefficient[pos_b])) * similarity[pos_b]).sum()
#     loss = (loss_t + loss_b) / len(y)
#
#     return loss
#
# def calculate_decode_loss(x, recon_x, sa_f=0.9):
#     """
#
#     Args:
#         x:              原始的光谱向量
#         recon_x:        重建的光谱向量
#
#     Returns:            重建损失函数
#
#     """
#     dot_product = torch.mm(x, recon_x.T)
#     # 获取对角线元素
#     diagonal_elements = torch.diag(dot_product)
#     # torch.norm 函数来计算向量的范数，默认计算二范数
#     norm_product = torch.norm(x, dim=1) * torch.norm(recon_x, dim=1)
#     spectral_cosine = diagonal_elements / norm_product
#     # torch.clamp(num, -1, 1) 将 num 限制在[-1, 1]区间内
#     spectral_cosine = torch.clamp(spectral_cosine, -0.999999, 0.999999)
#     spectral_angle = torch.acos(spectral_cosine)
#     loss_sa = torch.mean(spectral_angle / torch.pi)
#
#     # 添加 MSE 损失
#     mse_loss_fn = torch.nn.MSELoss()
#     loss_mse = mse_loss_fn(x, recon_x)
#
#     # 把两个损失组合起来
#     loss = sa_f * loss_sa + (1 - sa_f) * loss_mse
#
#     return loss, loss_sa, loss_mse
#
#
# def calculate_sparse_loss(hidden_activations, target_sparsity=0.01, beta=0.1):
#     """
#     计算稀疏性损失
#
#     Args:
#         hidden_activations: 隐藏层的激活值
#         target_sparsity: 目标稀疏性水平
#         beta: 控制稀疏性惩罚项的重要性
#
#     Returns:
#         sparsity_loss: 稀疏性损失
#     """
#     avg_activation = torch.mean(hidden_activations, dim=0)  # 计算每个隐藏单元的平均激活值
#     # 确保平均激活值在 [0, 1] 范围内
#     avg_activation = torch.clamp(avg_activation, min=0.000001, max=0.999999)
#     temp1 = torch.log(target_sparsity / avg_activation)
#     temp2 = torch.log((1 - target_sparsity) / (1 - avg_activation))
#     kl_divergence = target_sparsity * temp1 + (1 - target_sparsity) * temp2
#     sparsity_loss = beta * torch.sum(kl_divergence)
#     return sparsity_loss
#
# def calculate_regularization_loss(model, weight_decay=0.001):
#     """
#     计算权重衰减正则化损失
#
#     Args:
#         model: PyTorch模型
#         weight_decay: 权重衰减系数
#
#     Returns:
#         regularization_loss: 正则化损失
#     """
#     regularization_loss = 0
#     for param in model.parameters():
#         regularization_loss += torch.sum(param ** 2)
#     return weight_decay * regularization_loss
#
# def calculate_loss(x, encoded_x, encoded_ts, recon_x, y,
#             similarity_coefficient, model,
#             encode_f=0.5, alaf=0.2, metric='SA', beta=0.1, weight_decay=0.001):
#
#     # 计算编码器和解码器损失
#     encoded_loss = calculate_encode_loss(encoded_x, encoded_ts, y, similarity_coefficient, alaf, metric)
#     decoded_loss, decoded_loss_sa, decoded_loss_mse = calculate_decode_loss(x, recon_x)
#
#     # 加入稀疏性约束
#     sparsity_loss = calculate_sparse_loss(model.hidden_activations, beta=beta)
#
#     # 加入正则化项
#     regularization_loss = calculate_regularization_loss(model, weight_decay)
#
#     # 结合所有损失
#     ae_loss = encode_f * encoded_loss + (1 - encode_f) * decoded_loss
#     # ae_loss = encode_f * encoded_loss + (1 - encode_f) * decoded_loss + regularization_loss + sparsity_loss
#
#     return ae_loss, encoded_loss, decoded_loss
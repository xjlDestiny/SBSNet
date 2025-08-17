
import multiprocessing

import torch
import torch.nn as nn


class CEM():

    def __init__(self):
        self.name = 'CEM'

    def __call__(self, hsi_tensor: torch.Tensor, ts_tensor: torch.Tensor) -> torch.Tensor:
        # Basic implementation of the Constrained Energy Minimization (CEM) detector
        # Farrand, William H., and Joseph C. Harsanyi. "Mapping the distribution of mine tailings
        # in the Coeur d'Alene River Valley, Idaho, through the use of a constrained energy minimization
        # technique." Remote Sensing of Environment 59, no. 1 (1997): 64-76.
        size = hsi_tensor.shape  # get the size of image matrix
        R = torch.mm(hsi_tensor.T, hsi_tensor) / size[0]  # R = X*X'/size(X,2);
        w = torch.mm(ts_tensor, torch.inverse(R))  # w = (R+lamda*eye(size(X,1)))\t;
        detection_map = torch.mm(hsi_tensor, w.T)  # y=w'* X;
        return detection_map.squeeze()

class ACE():

    def __init__(self):
        self.name = 'ACE'

    def __call__(self, hsi_tensor: torch.Tensor, ts_tensor: torch.Tensor) -> torch.Tensor:
        # Basic implementation of the Adaptive Coherence/Cosine Estimator (ACE)
        # Manolakis, Dimitris, David Marden, and Gary A. Shaw.
        # "Hyperspectral image processing for automatic target detection applications."
        # Lincoln laboratory journal 14, no. 1 (2003): 79-116.
        size = hsi_tensor.shape
        img_mean = torch.mean(hsi_tensor, dim=0).unsqueeze(0)
        img0 = hsi_tensor - torch.matmul(torch.ones((size[0], 1), dtype=torch.float64).to(hsi_tensor.device), img_mean)
        R = torch.matmul(img0.t(), img0) / size[0]
        inv_R = torch.linalg.inv(R)
        diff_ts = ts_tensor - img_mean
        # 前两项y0, y1的计算无需变化
        y0 = torch.matmul(torch.matmul(diff_ts, inv_R), img0.t()).pow(2)
        y1 = torch.matmul(diff_ts, torch.matmul(inv_R, diff_ts.t()))
        # y2的计算需要进行轻微的改变
        y2 = torch.sum(torch.matmul(img0, inv_R) * img0, dim=1).unsqueeze(1)
        result = y0.t() / (y1 * y2)
        return result.squeeze()

class MF():

    def __init__(self):
        self.name = 'MF'

    def __call__(self, hsi_tensor: torch.Tensor, ts_tensor: torch.Tensor) -> torch.Tensor:
        # Basic implementation of the Matched Filter (MF)
        # Manolakis, Dimitris, Ronald Lockwood, Thomas Cooley, and John Jacobson. "Is there a best hyperspectral
        # detection algorithm?." In Algorithms and technologies for multispectral, hyperspectral, and ultraspectral
        # imagery XV, vol. 7334, p. 733402. International Society for Optics and Photonics, 2009.
        size = hsi_tensor.shape
        a = torch.mean(hsi_tensor, dim=0).unsqueeze(0)
        k = torch.matmul((hsi_tensor - a).t(), (hsi_tensor - a)) / size[0]
        w = torch.matmul(torch.linalg.inv(k), (ts_tensor - a).t())
        result = torch.matmul(hsi_tensor - a, w)
        return result.squeeze()

class OSP():

    def __init__(self):
        self.name = 'OSP'

    def __call__(self, hsi_tensor: torch.Tensor, ts_tensor: torch.Tensor) -> torch.Tensor:
        # Basic implementation of the Orthogonal Subspace Projection (OSP) approach
        # Zhong, Yanfei, and Liangpei Zhang. "An orthogonal subspace projection-based approach to H-dome transformation
        # for multi-source remote-sensing image matching." IEEE Transactions on Geoscience and Remote Sensing 54, no. 7 (2016): 4066-4078.
        size = hsi_tensor.shape
        R = torch.mm(hsi_tensor.T, hsi_tensor) / size[0]  # Compute correlation matrix
        P = torch.mm(torch.inverse(R), ts_tensor.T)  # Compute projection matrix P
        I = torch.eye(hsi_tensor.shape[1]).to(hsi_tensor.device)  # Identity matrix
        M = I - torch.mm(P, ts_tensor)  # Compute the projector onto the orthogonal complement of the target
        hsi_tensor_transformed = torch.mm(hsi_tensor, M)  # Project data using M
        detection_map = torch.norm(hsi_tensor_transformed, dim=1)  # Compute the norm
        return detection_map.squeeze()

class ECEM():

    def __init__(self):
        self.name = 'E_CEM'
        self.img = None
        self.tgt = None
        self.windowsize = [1/4, 2/4, 3/4, 4/4]
        self.num_layer = 10
        self.num_cem = 6
        self.Lambda = 1e-6
        self.show_proc = True

    def parmset(self, **parm):
        self.windowsize = parm['windowsize']
        self.num_layer = parm['num_layer']
        self.num_cem = parm['num_cem']
        self.Lambda = parm['Lambda']
        self.show_proc = parm['show_proc']

    def dual_sigmoid(self, x):
        weights = 1.0 / (1.0 + torch.exp(-x))
        return weights

    def cem(self, img, tgt):
        size = img.shape
        lamda = torch.rand((1,)).item()*(self.Lambda / (1 + self.Lambda)) + self.Lambda
        R = img @ img.t() / size[1]
        w = torch.linalg.inv(R + lamda * torch.eye(size[0])) @ tgt
        result = w.t() @ img
        return result

    def ms_scanning_unit(self, windowsize):
        # print('windowsize={} start'.format(windowsize))
        d = self.img.shape[0]
        winlen = int(d * windowsize ** 2)
        size = self.imgt.shape
        result = torch.zeros((int((size[0] - winlen + 1) / 2 + 1), size[1]))
        pos = 0
        for i in range(0, size[0] - winlen + 1, 2):
            imgt_tmp = self.imgt[i:i + winlen - 1, :]
            result[pos, :] = self.cem(imgt_tmp, imgt_tmp[:, -1])
            pos += 1
        # print('windowsize={} end'.format(windowsize))
        return result

    def cascade_detection(self, mssimg):
        size = mssimg.shape
        result_forest = torch.zeros((self.num_cem, size[1]))
        for i_layer in range(self.num_layer):
            for i_num in range(self.num_cem):
                result_forest[i_num, :] = self.cem(mssimg, mssimg[:, -1])
            weights = self.dual_sigmoid(torch.mean(result_forest, dim=0))
            mssimg *= weights
        result = result_forest[:, 0:-1]
        return result

    def __call__(self, hsi_tensor: torch.Tensor,
                ts_tensor: torch.Tensor, pool_num: int = 4) -> torch.Tensor:
        self.img = hsi_tensor.t()
        self.tgt = ts_tensor.t()
        self.imgt = torch.hstack((self.img, self.tgt))

        # 用一个简单的循环替换 multiprocessing.Pool
        results = []
        for windowsize in self.windowsize:
            result = self.ms_scanning_unit(windowsize)
            results.append(result)

        mssimg = torch.vstack(results)
        cadeimg = self.cascade_detection(mssimg)
        result = torch.mean(cadeimg, dim=0)[:self.imgt.shape[1]].unsqueeze(dim=-1)
        return result.squeeze()

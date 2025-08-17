import numpy as np
import random
import multiprocessing

class SAM():
    def __init__(self):
        self.name = 'SAM'

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray,
                           metric: str = 'Angle') -> np.ndarray:
        """
            X: type--tensor
                shape--HW*B
            t: type--tensor
                shape--1*B
        """
        # Spectral Angle
        dot_product = np.dot(hsi_ndarray, ts_ndarray.T)
        # 使用 np.linalg.norm 函数来计算矩阵的范数
        norm_product = np.linalg.norm(hsi_ndarray, axis=1) * np.linalg.norm(ts_ndarray)
        # 防止除零错误，确保norm_product非零
        norm_product[norm_product == 0] = 1e-5
        # 计算cosine similarity
        spectral_cosine = dot_product[:, 0] / norm_product
        # 将 spectral_cosince 限制在 [-1, 1] 区间内
        spectral_cosine = np.clip(spectral_cosine, -0.999999, 0.999999)
        # 计算光谱角度
        spectral_angle = np.arccos(spectral_cosine)
        # 防止除零错误，确保spectral_angle非零
        spectral_angle[spectral_angle == 0] = 1e-5
        # 计算相似度
        similarity = 1. / spectral_angle
        return similarity

class SA():
    def __init__(self):
        self.name = 'SA'

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray,
                           metric: str = 'Angle') -> np.ndarray:
        """
            X: type--tensor
                shape--HW*B
            t: type--tensor
                shape--1*B
        """
        # Spectral Angle
        dot_product = np.dot(hsi_ndarray, ts_ndarray.T)
        # 使用 np.linalg.norm 函数来计算矩阵的范数
        norm_product = np.linalg.norm(hsi_ndarray, axis=1) * np.linalg.norm(ts_ndarray)
        # 计算cosine similarity
        spectral_cosine = dot_product[:, 0] / norm_product
        # 防止除零错误，确保norm_product非零
        norm_product[norm_product == 0] = 1e-5
        # 将 spectral_cosince 限制在 [-1, 1] 区间内
        spectral_cosine = np.clip(spectral_cosine, -0.999999, 0.999999)
        # 计算光谱角度
        spectral_angle = np.arccos(spectral_cosine)
        # 计算相似度
        similarity = 1 - spectral_angle / np.pi
        return similarity

class ED():
    def __init__(self):
        self.name = 'ED'

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray) -> np.ndarray:
        """
            X: type--tensor
                shape--HW*B
            t: type--tensor
                shape--1*B
        """
        # Euclidean Distance
        euclidean_distance = np.sqrt(np.sum(np.square(hsi_ndarray - ts_ndarray), axis=1))
        # Calculate similarity, smaller distance means higher similarity
        similarity = np.exp(-euclidean_distance)
        return similarity

class CEM():

    def __init__(self):
        self.name = 'CEM'

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray) -> np.ndarray:
        # Basic implementation of the Constrained Energy Minimization (CEM) detector
        # Farrand, William H., and Joseph C. Harsanyi. "Mapping the distribution of mine tailings
        # in the Coeur d'Alene River Valley, Idaho, through the use of a constrained energy minimization
        # technique." Remote Sensing of Environment 59, no. 1 (1997): 64-76.
        size = hsi_ndarray.shape
        R = np.matmul(hsi_ndarray.T, hsi_ndarray) / size[0]
        Rinv_t = np.matmul(ts_ndarray, np.linalg.inv(R))
        w = Rinv_t / np.matmul(Rinv_t, ts_ndarray.T).item()
        detection_map = np.matmul(hsi_ndarray, w.T)
        return detection_map.squeeze()


class ACE():

    def __init__(self):
        self.name = 'ACE'

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray) -> np.ndarray:
        size = hsi_ndarray.shape
        img_mean = np.mean(hsi_ndarray, axis=0, keepdims=True)
        img0 = hsi_ndarray - np.matmul(np.ones((size[0], 1)), img_mean)
        diff_ts = ts_ndarray - img_mean
        R = np.matmul(img0.T, img0) / size[0]
        inv_R = np.linalg.inv(R)
        y0 = np.matmul(np.matmul(diff_ts, inv_R), img0.T) ** 2
        y1 = np.matmul(diff_ts, np.matmul(inv_R, diff_ts.T))
        y2 = np.sum(np.matmul(img0, inv_R) * img0, axis=1, keepdims=True)
        result = y0.T / (y1 * y2)
        return result.squeeze()



class MF():

    def __init__(self):
        self.name = 'MF'

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray) -> np.ndarray:
        size = hsi_ndarray.shape
        a = np.mean(hsi_ndarray, axis=0, keepdims=True)
        k = np.matmul((hsi_ndarray - a).T, (hsi_ndarray - a)) / size[0]
        w = np.matmul(np.linalg.pinv(k), (ts_ndarray - a).T)
        result = np.matmul(hsi_ndarray - a, w)
        return result.squeeze()


class OSP():

    def __init__(self):
        self.name = 'OSP'

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray) -> np.ndarray:
        size = hsi_ndarray.shape
        R = np.matmul(hsi_ndarray.T, hsi_ndarray) / size[0]
        P = np.matmul(np.linalg.pinv(R), ts_ndarray.T)
        I = np.eye(hsi_ndarray.shape[1])
        M = I - np.matmul(P, ts_ndarray)
        hsi_ndarray_transformed = np.matmul(hsi_ndarray, M)
        detection_map = np.linalg.norm(hsi_ndarray_transformed, axis=1)
        return detection_map.squeeze()


class ECEM():

    def __init__(self):
        self.name = 'E-CEM'
        self.windowsize = [1 / 4, 2 / 4, 3 / 4, 4 / 4]  # window size
        self.num_layer = 4  # the number of detection layers
        self.num_cem = 4  # the number of CEMs per layer
        self.Lambda = 1e-6  # the regularization coefficient
        self.show_proc = True  # show the process or not

    def parmset(self, **parm):
        self.windowsize = parm['windowsize']  # parameters
        self.num_layer = parm['num_layer']
        self.num_cem = parm['num_cem']
        self.Lambda = parm['Lambda']
        self.show_proc = parm['show_proc']

    def dual_sigmoid(self, x):
        x = np.array(x)
        weights = 1.0 / (1.0 + np.exp(-x))
        return weights

    def cem(self, img, tgt):
        size = img.shape  # get the size of image matrix
        lamda = random.uniform(self.Lambda / (1 + self.Lambda), self.Lambda)  # random regularization coefficient
        R = np.dot(img, img.T / size[1])  # R = X*X'/size(X,2);
        w = np.dot(np.linalg.inv((R + lamda * np.identity(size[0]))), tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
        result = np.dot(w.T, img)  # y=w'* X;
        return result

    def ms_scanning_unit(self, winowsize):
        d = self.img.shape[0]
        winlen = int(d * winowsize ** 2)
        size = self.imgt.shape  # get the size of image matrix
        result = np.zeros(shape=(int((size[0] - winlen + 1) / 2 + 1), size[1]))
        pos = 0
        # show process
        # if self.show_proc: print('Multi_Scale Scanning: size of window: %d' % winlen)
        for i in range(0, size[0] - winlen + 1, 2):  # multi-scale scanning
            imgt_tmp = self.imgt[i:i + winlen - 1, :]
            result[pos, :] = self.cem(imgt_tmp, imgt_tmp[:, -1])
            pos += 1
        return result

    def cascade_detection(self, mssimg):  # defult parameter configuration
        size = mssimg.shape
        result_forest = np.zeros(shape=(self.num_cem, size[1]))
        for i_layer in range(self.num_layer):
            if self.show_proc: print('Cascaded Detection layer: %d' % i_layer)  # show the process of cascade detection
            # show process
            # if self.show_proc: print('Cascaded Detection layer: %d' % i_layer)  # show the process of cascade detection
            for i_num in range(self.num_cem):
                result_forest[i_num, :] = self.cem(mssimg, mssimg[:, -1])
            weights = self.dual_sigmoid(np.mean(result_forest, axis=0))  # sigmoid nonlinear function
            mssimg = mssimg * weights
        result = result_forest[:, 0:-1]
        return result

    def __call__(self, hsi_ndarray: np.ndarray, ts_ndarray: np.ndarray) -> np.ndarray:
        self.img = hsi_ndarray.T
        self.tgt = ts_ndarray.T
        self.imgt = np.hstack((self.img, self.tgt))
        pool_num = len(self.windowsize)

        # results = []
        # for windowsize in self.windowsize:
        #     result = self.ms_scanning_unit(windowsize)
        #     results.append(result)
        p = multiprocessing.Pool(pool_num)  # multiprocessing
        results = p.map(self.ms_scanning_unit, self.windowsize)  # Multi_Scale Scanning
        p.close()
        p.join()

        mssimg = np.concatenate(results, axis=0)
        cadeimg = self.cascade_detection(mssimg)  # Cascaded Detection
        result = np.mean(cadeimg, axis=0)[:self.imgt.shape[1]].reshape(-1, 1)
        return result.squeeze()

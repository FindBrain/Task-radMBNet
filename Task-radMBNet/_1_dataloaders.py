# 导入使用到的库
import csv, os, random
import numpy as np
import SimpleITK as sitk
import torch
# from torchvision import transforms
import torch.utils.data as Data
# from absl import app, flags, logging
# import pandas as pd
import numpy as np
import scipy.io
import networkx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from net5_Attention_Net import Attention_Net

# 调试时的参数加载


'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要加载 __init__，__len__和__getitem__方法
'''

# 为训练数据 创建数据加载子类
class train_Dataset(Data.Dataset):
    def __init__(self, root, fold):
        super(train_Dataset, self).__init__()
        # 初始化
        self.root = root
        self.fold = fold
        # self.i=i

        # 通过CSV的形式获取图像与标签
        # self.images, self.labels = self.load_csv(str(self.fold) + '_train.csv')
        # 再次打乱数据引入随机性
        baselinemat, feamat, baselineimage,linear,labels = self.load_csv(str(self.fold) + '_train.csv')
        pairs = [[a, b, c,d,e] for a, b, c,d,e in zip(baselinemat, feamat,baselineimage,linear,labels)]
        random.shuffle(pairs)
        self.baselinemat = [i[0] for i in pairs]
        self.feamat = [i[1] for i in pairs]
        self.baselineimage=[i[2] for i in pairs]
        self.linear = [i[3] for i in pairs]
        self.labels = [i[4] for i in pairs]

    # 将所有的图像与标签路径配对好写入CSV
    def load_csv(self, filename):
        baselinemat, feamat,baselineimage,linear ,labels = [], [], [],[],[]
        with open(os.path.join(self.root, str(self.fold), filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                basemat, fmat,baseimage, linear1,label = row[0], row[1], row[2],row[3],row[4]
                label = int(label)
                baselinemat.append(basemat)
                baselineimage.append(baseimage)
                feamat.append(fmat)
                linear.append(linear1)
                labels.append(label)
        assert len(labels) == len(baselinemat) == len(feamat)==len(baselineimage)==len(linear)
        return baselinemat, feamat, baselineimage,linear,labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.baselinemat)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        baselinemat, feamat, baselineimage,linear,label = self.baselinemat[index], self.feamat[index],self.baselineimage[index], self.linear[index],self.labels[index]

        # i=self.i
        # 归一化
        def normalization(image_array):
            min = np.min(image_array)
            max = np.max(image_array)
            normalized_image_array = (image_array - min) / (max - min)
            return normalized_image_array

        # Nifti图像加载
        def str2img(x1):
            x1 = sitk.ReadImage(x1)
            x1 = sitk.GetArrayFromImage(x1)
            # subject_label_path = './xixi/aal109.nii'
            # label_array = sitk.GetArrayFromImage(sitk.ReadImage(subject_label_path))
            # label_array[label_array != i] = 0
            # x = x * label_array
            # x1 = normalization(x1)  # 对单张图像进行归一化
            x1 = np.pad(x1, ((18, 18), (9, 9), (18, 18)), 'constant', constant_values=0)
            x1 = torch.tensor(x1)
            x1 = x1.unsqueeze(0).float()

            return x1

        def buildmat(x1, x2):
            feanet_baseline = scipy.io.loadmat(x2)
            featopology_baseline = feanet_baseline['baseline_feature']
            img_baseline = abs(np.array(featopology_baseline))
            img_baseline = torch.tensor(img_baseline).float()

            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            topology = abs(network['average_pearson'])
            # topology = network['average_pearson']
            # topology[topology < 0.8] = 0
            topology = torch.tensor(topology).float()
            # topology = (topology - topology.min()) / (topology.max() - topology.min())
            # binarize topology
            # topology = (topology > 0.8).astype(float)

            return img_baseline, topology

        def buildmat_topo(x1):


            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            topology = abs(network['average_pearson'])
            # topology[topology < 0.5] = 0
            # topology = (topology - topology.min()) / (topology.max() - topology.min())
            # binarize topology
            # topology = (topology > 0.8).astype(float)
            topology = torch.tensor(topology).float()

            return  topology

        def normalize(mat, y, topology):
            data = torch_geometric.utils.from_networkx(networkx.convert_matrix.from_numpy_matrix(topology))
            data.x = torch.tensor(mat).float()
            data.y = torch.tensor(y).long()
            # data.cov = torch.tensor(cov_train[i, :]).float()
            return data
        def builtlinear(x1):
            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            topology = network['linear']
            topology[topology < 0.5] = 0
            # topology = (topology > 0.5).astype(float)
            # topology = normalization(topology)
            nettopology = torch.tensor(topology).float()
            # topology = topology.unsqueeze(0).float()
            # topology = (topology - topology.min()) / (topology.max() - topology.min())
            # binarize topology
            # topology = (topology > 0.9).astype(float)
            return nettopology
        image=str2img(baselineimage)
        mat, topology_pcc = buildmat(baselinemat, feamat)
        topology_topo = buildmat_topo(baselinemat)
        y = label
        # Graph = normalize(mat, y, topology)
        linears = builtlinear(linear)
        # 标签加载
        label = torch.tensor(label)

        return mat,topology_pcc,topology_topo,image,linears,label




# 为验证数据 创建数据加载子类
class val_Dataset(Data.Dataset):
    def __init__(self, root, fold):
        super(val_Dataset, self).__init__()
        # 初始化
        self.root = root
        self.fold = fold
        # self.i=i

        # 通过CSV的形式获取图像与标签
        # self.images, self.labels = self.load_csv(str(self.fold) + '_train.csv')
        # 再次打乱数据引入随机性
        baselinemat, feamat, baselineimage,linear,labels = self.load_csv(str(self.fold) + '_val.csv')
        pairs = [[a, b, c,d,e] for a, b, c,d,e in zip(baselinemat, feamat,baselineimage,linear,labels)]
        random.shuffle(pairs)
        self.baselinemat = [i[0] for i in pairs]
        self.feamat = [i[1] for i in pairs]
        self.baselineimage=[i[2] for i in pairs]
        self.linear = [i[3] for i in pairs]
        self.labels = [i[4] for i in pairs]

    # 将所有的图像与标签路径配对好写入CSV
    def load_csv(self, filename):
        baselinemat, feamat,baselineimage,linear ,labels = [], [], [],[],[]
        with open(os.path.join(self.root, str(self.fold), filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                basemat, fmat,baseimage, linear1,label = row[0], row[1], row[2],row[3],row[4]
                label = int(label)
                baselinemat.append(basemat)
                baselineimage.append(baseimage)
                feamat.append(fmat)
                linear.append(linear1)
                labels.append(label)
        assert len(labels) == len(baselinemat) == len(feamat)==len(baselineimage)==len(linear)
        return baselinemat, feamat, baselineimage,linear,labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.baselinemat)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        baselinemat, feamat, baselineimage,linear,label = self.baselinemat[index], self.feamat[index],self.baselineimage[index], self.linear[index],self.labels[index]

        # i=self.i
        # 归一化
        def normalization(image_array):
            min = np.min(image_array)
            max = np.max(image_array)
            normalized_image_array = (image_array - min) / (max - min)
            return normalized_image_array

        # Nifti图像加载
        def str2img(x1):
            x1 = np.pad(x1, ((18, 18), (9, 9), (18, 18)), 'constant', constant_values=0)
            x1 = torch.tensor(x1)
            x1 = x1.unsqueeze(0).float()

            return x1

        def buildmat(x1, x2):
            feanet_baseline = scipy.io.loadmat(x2)
            featopology_baseline = feanet_baseline['baseline_feature']
            img_baseline = abs(np.array(featopology_baseline))
            img_baseline = torch.tensor(img_baseline).float()

            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            # topology = network['average_pearson']
            topology = abs(network['average_pearson'])
            topology = torch.tensor(topology).float()
            return img_baseline, topology

        def buildmat_topo(x1):


            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            topology = abs(network['average_pearson'])
            topology = torch.tensor(topology).float()

            return  topology

        def normalize(mat, y, topology):
            data = torch_geometric.utils.from_networkx(networkx.convert_matrix.from_numpy_matrix(topology))
            data.x = torch.tensor(mat).float()
            data.y = torch.tensor(y).long()
            return data
        def builtlinear(x1):
            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            topology = network['linear']
            topology[topology < 0.5] = 0
            nettopology = torch.tensor(topology).float()
            return nettopology
        image=str2img(baselineimage)
        mat, topology_pcc = buildmat(baselinemat, feamat)
        topology_topo = buildmat_topo(baselinemat)
        y = label
        # Graph = normalize(mat, y, topology)
        linears = builtlinear(linear)
        # 标签加载
        label = torch.tensor(label)

        return mat,topology_pcc,topology_topo,image,linears,label

# 为测试数据 创建数据加载子类
class test_Dataset(Data.Dataset):
    def __init__(self, root, fold):
        super(test_Dataset, self).__init__()
        # 初始化
        self.root = root
        self.fold = fold
        baselinemat, feamat, baselineimage,linear,labels = self.load_csv(str(self.fold) + '_test.csv')
        pairs = [[a, b, c,d,e] for a, b, c,d,e in zip(baselinemat, feamat,baselineimage,linear,labels)]
        random.shuffle(pairs)
        self.baselinemat = [i[0] for i in pairs]
        self.feamat = [i[1] for i in pairs]
        self.baselineimage=[i[2] for i in pairs]
        self.linear = [i[3] for i in pairs]
        self.labels = [i[4] for i in pairs]

    # 将所有的图像与标签路径配对好写入CSV
    def load_csv(self, filename):
        baselinemat, feamat,baselineimage,linear ,labels = [], [], [],[],[]
        with open(os.path.join(self.root, str(self.fold), filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                basemat, fmat,baseimage, linear1,label = row[0], row[1], row[2],row[3],row[4]
                label = int(label)
                baselinemat.append(basemat)
                baselineimage.append(baseimage)
                feamat.append(fmat)
                linear.append(linear1)
                labels.append(label)
        assert len(labels) == len(baselinemat) == len(feamat)==len(baselineimage)==len(linear)
        return baselinemat, feamat, baselineimage,linear,labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.baselinemat)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        baselinemat, feamat, baselineimage,linear,label = self.baselinemat[index], self.feamat[index],self.baselineimage[index], self.linear[index],self.labels[index]

        def normalization(image_array):
            min = np.min(image_array)
            max = np.max(image_array)
            normalized_image_array = (image_array - min) / (max - min)
            return normalized_image_array

        # Nifti图像加载
        def str2img(x1):
            x1 = sitk.ReadImage(x1)
            x1 = sitk.GetArrayFromImage(x1)
            x1 = np.pad(x1, ((18, 18), (9, 9), (18, 18)), 'constant', constant_values=0)
            x1 = torch.tensor(x1)
            x1 = x1.unsqueeze(0).float()

            return x1

        def buildmat(x1, x2):
            feanet_baseline = scipy.io.loadmat(x2)
            featopology_baseline = feanet_baseline['baseline_feature']
            img_baseline = abs(np.array(featopology_baseline))
            img_baseline = torch.tensor(img_baseline).float()

            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            # topology = network['average_pearson']
            topology = abs(network['average_pearson'])
            topology = torch.tensor(topology).float()
            return img_baseline, topology

        def buildmat_topo(x1):


            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            topology = abs(network['average_pearson'])
            topology = torch.tensor(topology).float()

            return  topology

        def normalize(mat, y, topology):
            data = torch_geometric.utils.from_networkx(networkx.convert_matrix.from_numpy_matrix(topology))
            data.x = torch.tensor(mat).float()
            data.y = torch.tensor(y).long()
            return data
        def builtlinear(x1):
            net_mat_path = x1
            network = scipy.io.loadmat(net_mat_path)
            topology = network['linear']
            topology[topology < 0.5] = 0
            nettopology = torch.tensor(topology).float()
            return nettopology
        image=str2img(baselineimage)
        mat, topology_pcc = buildmat(baselinemat, feamat)
        topology_topo = buildmat_topo(baselinemat)
        y = label
        linears = builtlinear(linear)
        # 标签加载
        label = torch.tensor(label)

        return mat,topology_pcc,topology_topo,image,linears,label

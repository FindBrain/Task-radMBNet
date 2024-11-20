import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



def matrix_top(data_feature,device):
    data_feature = data_feature.detach().cpu().numpy()
    batches = data_feature.shape[0]
    list = np.zeros((batches,90,90))
    for i in range(batches):
        matx = np.zeros((90,90))
        feature = abs(data_feature[i])
        n = feature.shape[0]
        indices = np.triu_indices(n,k=1)
        feature1 = feature[indices]
        row_means = np.mean(feature1)
        feature1[feature1<row_means]=0
        matx[indices] = feature1
        matx = matx+matx.T
        # corr = np.corrcoef(feature)
        # list[i] = feature[indices]
        list[i] = matx
    feature_matrix = torch.tensor(list).float().to(device)
    return  feature_matrix


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        # support = torch.mm(input_feature, self.weight)
        support = torch.matmul(input_feature, self.weight)
        # output = torch.sparse.mm(adjacency, support)
        output = torch.einsum('bij,bjd->bid', [adjacency, support])
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


# out1_feature, out2_feature, out3_feature, dropout
class GCNGenerator(nn.Module):
    def __init__(self, in_feature, out_feature, nodes):
        super(GCNGenerator, self).__init__()



        self.gc1_01 = GraphConvolution(in_feature, int(in_feature * 2))
        self.LayerNorm1_01 = nn.LayerNorm([nodes, int(in_feature * 2)])
        self.gc1_12 = GraphConvolution(int(in_feature * 2), out_feature)
        self.LayerNorm1_12 = nn.LayerNorm([nodes, out_feature])

        self.gc2_01 =GraphConvolution(in_feature, int(in_feature * 2))
        self.LayerNorm2_01 = nn.LayerNorm([nodes, int(in_feature * 2)])
        self.gc2_12 = GraphConvolution(int(in_feature * 2), out_feature)
        self.LayerNorm2_12 = nn.LayerNorm([nodes, out_feature])

        self.gc3_01 = GraphConvolution(25, int(25 * 2))
        self.LayerNorm3_01 = nn.LayerNorm([nodes, int(25 * 2)])
        self.gc3_12 = GraphConvolution(int(25 * 2), 25)
        self.LayerNorm3_12 = nn.LayerNorm([nodes, 25])

        self.attention = nn.Conv2d(1,1,kernel_size=(1,90),stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.droup1 = nn.Dropout(0.5)
        self.droup2 = nn.Dropout(0.5)
        self.droup3 = nn.Dropout(0.5)

        self.dense11 = nn.Linear(2250, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dense22 = nn.Linear(512, 32)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(2250),  # 352
            nn.Linear(2250, 1)
        )

        self.sigmod = nn.Sigmoid()

        self.dense1 = nn.Linear(5, 25)
        self.relu = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(25, 32)
        self.droup = nn.Dropout(0.5)
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(32),  # 352
            nn.Linear(32, 1)
        )
        self.weight = nn.Parameter(torch.FloatTensor([0.0, 0.0]))


    def forward(self,nodemat, matpcc,device):



        topo = matpcc
        x2 = self.gc2_01(matpcc, topo)
        x2 = self.LayerNorm2_01(x2)
        x2 = F.leaky_relu(x2, 0.05, inplace=False)
        # x2 = x2.unsqueeze(1)
        x = matrix_top(x2, device)
        # a=x
        # x2 = self.droup2(x2)
        x2 = self.gc2_12(x,x2)
        x2 = self.LayerNorm2_12(x2)
        x2 = F.leaky_relu(x2, 0.05, inplace=False)
        x3 = self.gc3_01(x2,nodemat)
        # x3 = self.gc3_01(matrix_kl,nodemat)
        x3 = self.LayerNorm3_01(x3)
        x3 = F.leaky_relu(x3, 0.05, inplace=False)
        x3 = self.droup3(x3)
        x3 = self.gc3_12(x2,x3)
        x3 = self.LayerNorm3_12(x3)
        x3 = F.leaky_relu(x3, 0.05, inplace=False)
        x3 = torch.flatten(x3, start_dim=1)
        x3 = self.dense11(x3)
        x3 = self.relu(x3)
        x3 = self.dense22(x3)
        x3 = self.relu(x3)
        x3 = x3.unsqueeze(1)
        x3 = self.mlp_head1(x3)

        out = self.sigmod(x3)
        out = out.view(out.size(0))
        outputs = out

        return  outputs
















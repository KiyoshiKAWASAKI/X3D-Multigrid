import torch.nn as nn
import torch
from torch.nn.functional import normalize


class mlp_network(nn.Module):
    # TODO (Q): do we still need the resnet here? I don't think so?
    # def __init__(self, resnet, feature_dim, class_num):
    def __init__(self, feature_dim, class_num):
        super(mlp_network, self).__init__()
        # self.resnet = resnet
        # feature_dim: ???
        # cluster_num: equals to number of classes
        # resnet.rep_dim: 512 why??

        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.rep_dim = 26

        # self.instance_projector = nn.Sequential(
        #     nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.resnet.rep_dim, self.feature_dim),
        # )
        # self.cluster_projector = nn.Sequential(
        #     nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.resnet.rep_dim, self.cluster_num),
        #     nn.Softmax(dim=1)
        # )

        self.instance_projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.cluster_num),
            # nn.Softmax(dim=1)
        )
    # TODO: we do unsupervised learning partially
    def forward(self, x_i, x_j):
        # h_i = self.resnet(x_i)
        # h_j = self.resnet(x_j)

        # shape of original h_i and h_j: [256, 512]
        # shape of current h_i/x_i and h_j/x_j: [16, 26]

        z_i = normalize(self.instance_projector(x_i), dim=1)
        z_j = normalize(self.instance_projector(x_j), dim=1)

        c_i = self.cluster_projector(x_i)
        c_j = self.cluster_projector(x_j)

        # print(z_i)
        # print(z_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

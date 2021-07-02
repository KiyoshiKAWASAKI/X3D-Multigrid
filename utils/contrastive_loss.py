import torch
import torch.nn as nn
import math


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = None

        self.mask = self.mask_correlated_samples(batch_size)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum").to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="sum").cuda()

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    # def forward(self, z_i, z_j):
    #     N = 2 * self.batch_size
    #     z = torch.cat((z_i, z_j), dim=0)
    #
    #     sim = torch.matmul(z, z.T) / self.temperature
    #     sim_i_j = torch.diag(sim, self.batch_size)
    #     sim_j_i = torch.diag(sim, -self.batch_size)
    #
    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    #     negative_samples = sim[self.mask].reshape(N, -1)
    #
    #     labels = torch.zeros(N).to(positive_samples.device).long()
    #     logits = torch.cat((positive_samples, negative_samples), dim=1)
    #     loss = self.criterion(logits, labels)
    #     loss /= N
    #
    #     return loss

    def forward(self, z_i, z_j, labels):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels_fake = torch.zeros(N).to(positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        # print("@" * 20)
        # print(z_i.shape)
        # print(z_i.type(), z_i.dtype)
        #
        # print("labels")
        # print(labels.type(), labels.dtype)


        # torch.zeros((100, 100)).to(0)

        # z_i_cuda = z_i.to(labels.device)
        z_i_cuda = z_i.cuda() + 1e-10

        # print(z_i_cuda)
        # print(labels)
        #
        # print(z_i_cuda.shape, labels.shape)

        # loss_i = self.criterion(z_i_cuda, labels.long())
        loss_i = self.criterion(z_i_cuda, torch.max(labels, 1)[1])

        # z_j_cuda = z_j.to(labels.device)
        z_j_cuda = z_j.cuda() + 1e-10
        loss_j = self.criterion(z_j_cuda, torch.max(labels, 1)[1])

        # print(torch.max(labels, 1)[1])

        # z_i_cuda = z_i.to(positive_samples.device)
        # z_j_cuda = z_j.to(positive_samples.device)

        # loss_i = self.criterion(z_i_cuda, labels)
        # loss_j = self.criterion(z_j_cuda, labels)

        loss = (loss_i + loss_j)/N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum").cuda()
        self.criterion = nn.NLLLoss(reduction="sum").cuda()
        self.similarity_f = nn.CosineSimilarity(dim=2).cuda()

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        eps = 1e-7

        p_i = c_i.sum(0).view(-1)
        p_i = p_i / (p_i.sum() + eps)

        p_j = c_j.sum(0).view(-1)
        p_j = p_j / (p_j.sum() + eps)

        # for i in range(len(p_i)):
        #     if p_i[i] < 0:
        #         p_i[i] = -p_i[i]
        #
        # for i in range(len(p_j)):
        #     if p_j[i] < 0:
        #         p_j[i] = -p_j[i]

        s = nn.Softmax()

        p_i = s(p_i)
        p_j = s(p_j)

        # ne_i and ne_j are NAN...
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + eps)).sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + eps)).sum()

        # print("@" * 20)
        # print(p_i)
        # print(torch.log(p_i)) # TODO: this one has some NAN
        # print((p_i * torch.log(p_i)).sum())
        # print("@" * 20)

        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)

        # print(labels.type())
        # print(labels.type())

        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

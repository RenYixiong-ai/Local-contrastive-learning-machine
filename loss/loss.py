import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import math

class FBMLoss(nn.Module):
    def __init__(self, num_classes, lam, a, b=1.0, beta=50.0, if_onehot=False, losstype="fast_FermiBose"):
        super(FBMLoss, self).__init__()
        self.len_out = num_classes
        self.lam =lam
        self.a = a
        self.b = b
        self.beta = beta
        self.if_onehot = if_onehot
        self.losstype = losstype

    def forward(self, inputs, targets, weights=None):
        if weights is not None:
            normal = 0.5 * self.lam / self.len_out * torch.norm(weights, p=2) ** 2
        else:
            normal = 0

        if not self.if_onehot: targets = F.one_hot(targets).float()  #编码

        #损失类型
        if self.losstype == "fast_FermiBose":
            loss = fast_FermiBose(inputs, targets, self.a, self.b, self.beta) 
        elif self.losstype == "fast_StrongInter":
            loss = fast_StrongInter(inputs, targets, self.a, self.b) 
        
        #print(newfast_FermiBose(inputs, targets, self.d_f) - FermiBose(inputs, targets, self.d_f))
        return loss + normal

########################
##     FermiBose      ##
########################

def fast_FermiBose(sample, labels, d_f, alpha=1.0, beta=10.0):
    batch, hidden1_features = sample.shape
    # 使用广播计算每对样本之间的 L2 距离的平方和
    sample_diff = sample.unsqueeze(1) - sample.unsqueeze(0)  # 扩展维度并相减，得到 (batch, batch, outdim)
    D_matrix = torch.sum(sample_diff**2, dim=2)/hidden1_features  # 对最后一个维度求和，得到 (batch, batch) 矩阵. 对每一个数据进行归一化

    # 计算标签矩阵的乘积，结果是 (batch_size, batch_size)
    label_matrix = labels @ labels.T

    # 计算phi(.)
    diff_label_matrix = F.relu(d_f-D_matrix)     
    same_label_matrix = F.relu(D_matrix-d_f/beta)     # 用以固定Bose对的距离


    # 计算bose_loss
    bose_matrix = torch.mul(same_label_matrix, label_matrix)
    num_boson = torch.count_nonzero(bose_matrix)        # 计算Bose类别的数量
    bose_loss = torch.triu(bose_matrix, diagonal=1)
    bose_loss = bose_loss.sum()/num_boson               # 对统计的数目进行归一化

    # 计算fermi_loss
    fermi_matrix = torch.mul(diff_label_matrix, label_matrix)
    num_fermi = torch.count_nonzero(fermi_matrix)        # 计算Fermi类别的数量
    fermi_loss = torch.triu(fermi_matrix, diagonal=1)
    fermi_loss = fermi_loss.sum()/num_fermi              # 对统计的数目进行归一化

    # 总loss
    total_loss = bose_loss + alpha*fermi_loss

    return total_loss


#用循环写FermiBose，但是这两个有一些区别
def FermiBose(sample, labels, d_f):
    batch, hidden1_features = sample.shape
    loss = 0
    distance = nn.PairwiseDistance(p=2,keepdim=True)
    ReLU = nn.ReLU()

    cal_count = 1

    for i in range(batch):
        for j in range(i+1, batch):
            D = distance(sample[i], sample[j]) ** 2 / hidden1_features
            if labels[i] == labels[j]:
                loss += D.sum()
                #print("bose", D)
                if D > 0 : cal_count+=1
            else:
                loss += ReLU(d_f-D).sum()
                #print("fermi", D)
                if ReLU(d_f-D).sum() > 0.0: 
                    #print("dont, count", d_f, D, ReLU(d_f-D))
                    cal_count+=1
            
    return loss/cal_count

########################
##   强相互作用力损失    ##
########################

#用循环写这个
def StrongInter(sample, labels, a, b):
    """
    计算给定样本和标签的 Fermi-Bose 损失。

    参数:
    sample : torch.Tensor
        样本输入，形状为 (batch_size, hidden1_features)，其中 batch_size 是批大小，hidden1_features 是特征维度。
    labels : torch.Tensor
        样本对应的标签，形状为 (batch_size,)。
    a : float
        控制损失函数形状的参数，通常用于调整距离的非线性缩放。
    b : float
        控制损失函数形状的参数，与 a 配合定义距离的惩罚函数。

    返回:
    torch.Tensor
        计算得到的 Fermi-Bose 损失值。

    功能:
    - 计算样本之间的两两距离。
    - 对于标签相同的样本对，计算正损失。
    - 对于标签不同的样本对，计算负损失。
    - 返回标签相同和不同的样本对的平均损失之和。

    算法:
    - 损失函数基于以下公式:
      v(r) = a/r + b*r - sqrt(a*b)
      其中 r 是样本之间的欧几里得距离。
    - 标签相同的样本对会贡献 v(r)；
    - 标签不同的样本对会贡献 -v(r)。
    """
    batch, hidden1_features = sample.shape
    loss_same = 0
    loss_diff = 0
    distance = nn.PairwiseDistance(p=2,keepdim=True)

    same_label_count = 1
    diff_label_count = 1
    v = lambda x: a/(x + 1e-5)+b*x - math.sqrt(a*b)

    for i in range(batch):
        for j in range(i+1, batch):
            r = torch.norm(sample[i]-sample[j])
            if labels[i] == labels[j]:
                loss_same += v(r)
                same_label_count  += 1
            else:
                loss_diff += -v(r)
                diff_label_count  += 1
            #print("loss_loss", loss_same, loss_diff)
    print("loss", same_label_count, diff_label_count)
    print("loss", loss_same/same_label_count, loss_diff/diff_label_count)
    return loss_same/same_label_count + loss_diff/diff_label_count


def fast_StrongInter(sample, labels, a, b):
    batch, hidden1_features = sample.shape
    # 使用广播计算每对样本之间的 L2 距离的平方和
    sample_diff = sample.unsqueeze(1) - sample.unsqueeze(0)  # 扩展维度并相减，得到 (batch, batch, outdim)
    D_matrix = torch.norm(sample_diff, dim=2)# 对最后一个维度求和，得到 (batch, batch) 矩阵

    #mask = ~torch.eye(batch, dtype=bool, device=sample.device)  # 非对角线部分为 True

    # 计算phi(.)
    v = lambda x: a / (x+ 1e-5) + b * x - math.sqrt(a * b)
    phi_matrix = v(D_matrix) 

    # 计算标签矩阵的乘积，结果是 (batch_size, batch_size)，并且只保留上三角，去除重复以及自身关联项
    label_matrix = torch.triu(labels @ labels.T, diagonal=1).detach()

    # 计算bose_loss
    same_label_count = max(torch.sum(label_matrix), 1)
    bose_loss = torch.triu(label_matrix*phi_matrix, diagonal=1).sum()/same_label_count

    # 计算fermi_loss
    diff_label_count = max(batch*(batch-1)/2 - same_label_count, 1)
    fermi_loss = torch.triu((label_matrix-1)*phi_matrix, diagonal=1).sum()/diff_label_count

    # 总loss
    loss = bose_loss + fermi_loss
    #print("parallel", same_label_count, diff_label_count)
    #print("parallel", bose_loss, fermi_loss)
    return loss







def test_accuracy(model, test_loader, device):
    # 准确率计数
    correct = 0
    total = 0

    # 得到inout_size
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    batch, channel, large, _ = images.shape
    input_size = channel*large**2

    # 禁用梯度计算，加速测试过程
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据加载到 GPU
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            
            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            
            # 更新计数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 1.0 * correct / total
    #print(f'Accuracy on the test dataset: {accuracy:.2f}%') 

    return accuracy


if __name__ == "__main__":
    pass
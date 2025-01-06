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
    D_matrix = torch.sum(sample_diff**2, dim=2)/hidden1_features  # 对最后一个维度求和，得到 (batch, batch) 矩阵

    # 计算phi(.)
    phi_matrix = F.relu(d_f-D_matrix)
    bose_matrix = F.relu(D_matrix-d_f/beta)     # 用以固定Bose对的距离

    # 计算标签矩阵的乘积，结果是 (batch_size, batch_size)
    label_matrix = labels @ labels.T

    # 计算bose_loss
    #bose_loss = torch.mul(D_matrix, label_matrix)
    bose_loss = torch.mul(bose_matrix, label_matrix)

    # 计算fermi_loss
    fermi_loss = torch.mul(phi_matrix, 1-label_matrix)

    # 总loss
    loss_matrix = bose_loss + alpha*fermi_loss

    # 如果只需要上三角部分（排除重复项和自相似项）
    loss_matrix = torch.triu(loss_matrix, diagonal=1)
    cal_count = (loss_matrix > 0.0).sum().item()+1  #统计真正计算的fermi-bose对数量

    # 将结果进行 sum（可以选择性地只关注上三角部分的和）
    total_loss = loss_matrix.sum()
    #print(cal_count)
    return total_loss/cal_count
    #return 2*total_loss/batch**2

#用循环写这个
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
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50, resnet18,vgg16,vgg19

# 定义单层神经网络模型
class FBMLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FBMLayer, self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # 展平图像
        out = self.linear(x)
        #out = F.softmax(out, dim=-1)                # ！！！用于SILoss的激活函数，有问题
        out = (torch.tanh(out) + 1)/2.0    # 用于FermiBose的激活函数
        return out


# 定义全连接层神经网络模型
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # 展平图像
        out = self.linear(x)
        #out = F.softmax(out)
        return out


class ModelPipeline(nn.Module):
    def __init__(self, models=None):
        super(ModelPipeline, self).__init__()
        # 初始化模型列表
        self.models = nn.ModuleList(models) if models else nn.ModuleList()

    def add_model(self, model):
        """向管道中添加一个新模型"""
        self.models.append(model)

    def forward(self, x):
        """依次将输入数据通过所有模型"""
        for model in self.models:
            x = model(x)
        return x

class ParallelNetworks(nn.Module):
    def __init__(self, networks):
        """
        初始化多网络模块。

        参数:
            networks (list): 一个包含多个神经网络的列表，每个网络将接收相同的输入。
        """
        super(ParallelNetworks, self).__init__()
        self.networks = nn.ModuleList(networks)  # 将网络列表包装成 nn.ModuleList

    def forward(self, x):
        """
        前向传播，同时运行所有网络，并将结果拼接。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, ...]。

        返回:
            torch.Tensor: 拼接后的输出张量。
        """
        # 同时运行多个网络，并将它们的输出收集到一个列表中
        outputs = [network(x) for network in self.networks]
        # 沿着最后一维 (feature 维度) 拼接所有网络的输出
        return torch.cat(outputs, dim=1)

if __name__ == "__main__":
    pass
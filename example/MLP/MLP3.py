import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys

local_path = os.getcwd()
# 将项目主目录路径添加到 Python 路径
os.chdir("../../")  # 使用相对路径将工作目录切换到 project 文件夹
project_path = os.path.abspath(os.path.join(local_path, "../../"))
sys.path.append(project_path)   #将模块查找路径切换
local_path = os.path.join(os.getcwd(), "result/cluster")
os.makedirs(local_path, exist_ok=True)
print(local_path)

import torch
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from utils import *
set_seed(42)  # 42 是一个示例种子数，您可以根据需求更改


# 定义一个6层的全连接神经网络（MLP）
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # 定义6层全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)  # 第一层，输入28x28的图片，输出512维
        self.fc2 = nn.Linear(512, 256)      # 第二层，512维输入，输出256维
        self.fc3 = nn.Linear(256, 128)      # 第三层，256维输入，输出128维
        self.fc4 = nn.Linear(128, 64)       # 第四层，128维输入，输出64维
        self.fc5 = nn.Linear(64, 32)        # 第五层，64维输入，输出32维
        self.fc6 = nn.Linear(32, 10)        # 输出层，32维输入，输出10维（对应10个数字类别）

        # 激活函数
        self.relu = nn.ReLU()

        # 初始化每一层的参数为正态分布
        self._initialize_weights()

        # 冻结fc3层的参数，使其不参与训练
        for param in self.fc3.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x

    # 自定义的权重初始化方法
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 使用正态分布初始化权重
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 将偏置初始化为0

# 设置超参数
batch_size = 64
learning_rate = 0.001
epochs = 50

# 定义数据预处理和数据加载
class_counts = [100]*10
datatype = 'MNIST'
images_size = 1*28*28

train_loader = get_dataloader(datatype, batch_size=64, train=True, class_counts=class_counts)
test_loader = get_dataloader(datatype, batch_size=64, train=False)

# 实例化网络、损失函数和优化器
model = MLP()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于分类问题
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器

# 训练模型
for epoch in range(epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 计算训练集准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {100 * correct / total:.2f}%")

    # 每个epoch结束后计算一次测试集准确率
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度，节省内存和计算
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy after epoch {epoch+1}: {100 * correct / total:.2f}%")



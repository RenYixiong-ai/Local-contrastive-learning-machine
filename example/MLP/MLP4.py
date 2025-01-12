import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import os
import sys

local_path = os.getcwd()
# 将项目主目录路径添加到 Python 路径
os.chdir("../../")  # 使用相对路径将工作目录切换到 project 文件夹
project_path = os.path.abspath(os.path.join(local_path, "../../"))
sys.path.append(project_path)   #将模块查找路径切换
local_path = os.path.join(os.getcwd(), "result/MLP")
os.makedirs(local_path, exist_ok=True)
print(local_path)

import torch
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from utils import *
set_seed(42)  # 42 是一个示例种子数，您可以根据需求更改


# 定义一个6层的全连接神经网络（MLP）
# 定义一个多层感知机（MLP）模型，支持可变数量的中间层
class MLP(nn.Module):
    def __init__(self, num_hidden_layers):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        # 第一层是输入层，固定为28*28维
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        
        # 动态生成中间层
        layers = []
        in_features = 512
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, 512))
            layers.append(self.relu)
            in_features = 512
        
        # 合并所有中间层
        self.middle_layers = nn.Sequential(*layers)
        
        # 输出层，固定为10维（对应10个类别）
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)

        #self._initialize_weights()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.middle_layers(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
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
epochs = 500

# 定义数据预处理和数据加载
#class_counts = [100]*10
datatype = 'MNIST'
images_size = 1*28*28

train_loader = get_dataloader(datatype, batch_size=64, train=True) #, class_counts=class_counts)
test_loader = get_dataloader(datatype, batch_size=64, train=False)

# 定义训练和测试过程
def train_and_evaluate(num_hidden_layers):
    # 初始化模型，损失函数和优化器
    model = MLP(num_hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        # 在每个epoch之后评估测试集上的准确率
        model.eval()  # 设置模型为评估模式
        correct_test = 0
        total_test = 0

        with torch.no_grad():  # 不计算梯度
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        accuracy = 100 * correct_test / total_test
        best_accuracy = max(best_accuracy, accuracy)

    return best_accuracy

# 用于记录不同层数下的准确率
accuracies = []
num_hidden = 15

# 在不同层数下训练并评估
for num_layers in range(1, num_hidden):
    accuracy = train_and_evaluate(num_layers)
    accuracies.append(accuracy)
    print(f"Hidden Layers: {num_layers}, Best Accuracy: {accuracy:.2f}%")

# 绘制层数与准确率的关系图
plt.plot(range(1, num_hidden), accuracies, marker='o', linestyle='-', color='b')
plt.title("Accuracy vs Number of Hidden Layers")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, num_hidden))
plt.grid(True)
plt.savefig(os.path.join(local_path, "MultMLP_withoutNormal.png"))
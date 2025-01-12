import os
import sys

local_path = os.getcwd()

# 将项目主目录路径添加到 Python 路径
os.chdir("../../")  # 使用相对路径将工作目录切换到 project 文件夹
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_path)   #将模块查找路径切换

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import modelset
from train.train import train_FBM
from utils import *
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(42)

train_loader, test_loader = load_small_MNIST()
data_iter = iter(train_loader)
images, labels = next(data_iter)
batch, channel, large, _ = images.shape


def get_accuracy(d_f):
    # 定义超参数
    input_size = channel * large**2 # MNIST图像大小是28x28
    hidden_dim =1000
    num_classes = 10      # MNIST有10个类别
    learning_rate = 0.01
    lam = 0.01
    alpha = 1.0
    num_epochs = 20
    batch_size = 64

    model = train_FBM(input_size = input_size, 
                    output_size = hidden_dim, 
                    lam = lam, 
                    d_f = d_f, 
                    alpha = alpha,
                    learning_rate = learning_rate, 
                    train_loader = train_loader, 
                    num_epochs = num_epochs, 
                    device = device)

    train_loader1 = deal_dataloader(train_loader, model, device)



    # 定义损失函数和优化器
    model2 = modelset.MLP(hidden_dim, num_classes).to(device)
    criterion2 = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = optim.Adam(model2.parameters(), lr=learning_rate)  # 使用随机梯度下降优化器

    model2.train()
    # 训练模型
    epochs = 30
    for epoch in range(epochs):
        for images, labels in train_loader1:
            # 将图像展平为一维向量，并将标签进行 one-hot 编码
            images = images.to(device)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)  # 将标签转换为 one-hot 编码

            # 前向传播
            outputs = model2(images)

            # 计算损失
            loss = criterion2(outputs, labels_one_hot)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


    # 设置模型为评估模式
    model.eval()
    model2.eval()

    # 准确率计数
    correct = 0
    total = 0

    # 禁用梯度计算，加速测试过程
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据加载到 GPU
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            outputs = model2(outputs)
            
            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            
            # 更新计数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 100 * correct / total
    #print(f'Accuracy on the test dataset: {accuracy:.2f}%')
    return accuracy

import matplotlib.pyplot as plt
import pandas as pd

# 假设 d_f 是在一定范围内变化的
#d_f_values = [0.01 * i for i in range(200)]  # 生成 0 到 5 的 d_f 值
#a_values = [get_accuracy(d_f) for d_f in d_f_values]  # 计算对应的 a 值

# 多次采样求平均
d_f_values = [0.01 * i for i in range(100)]  # 生成 0 到 5 的 d_f 值
num_samples = 2  # 每个 d_f 值进行 10 次采样

# 对每个 d_f 进行多次采样并求平均
a_values = []
for d_f in d_f_values:
    sampled_accuracies = [get_accuracy(d_f) for _ in range(num_samples)]
    average_accuracy = np.mean(sampled_accuracies)  # 求平均
    a_values.append(average_accuracy)

    print('-'*30)
    print(d_f)
    print('-'*30)

# 绘制 a 随 d_f 变化的曲线
plt.figure(figsize=(8, 6))
plt.plot(d_f_values, a_values, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('d_f')
plt.ylabel('a (Accuracy)')
plt.title('Accuracy vs d_f')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(os.path.join(local_path, "accuracy_vs_d_f.png"))

# 保存数据到 CSV 文件
data = {'d_f': d_f_values, 'a': a_values}
df = pd.DataFrame(data)
df.to_csv(os.path.join(local_path, "accuracy_vs_d_f.csv"), index=False)

# 显示图像
#plt.show()


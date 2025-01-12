import os
import sys

local_path = os.getcwd()
# 将项目主目录路径添加到 Python 路径
os.chdir("../../")  # 使用相对路径将工作目录切换到 project 文件夹
project_path = os.path.abspath(os.path.join(local_path, "../../"))
sys.path.append(project_path)   #将模块查找路径切换
local_path = os.path.join(os.getcwd(), "result")
print(local_path)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import modelset
from train.train import train_FBM
from train.train import DFBM
from utils import *

import optuna
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(42)

train_loader, test_loader = load_small_cifar10()
#train_loader, test_loader = load_small_MNIST()
data_iter = iter(train_loader)
images, labels = next(data_iter)
batch, channel, large, _ = images.shape

csv_file_path = os.path.join(local_path, '66666.csv')

# 如果文件不存在，则创建文件并写入标题行
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['trial_number', 'output_size_list', 'd_f_list', 'alpha_list', 'accuracy'])

def objective(trial):
    # 定义超参数
    images_size = channel * large**2 # MNIST图像大小是28x28
    num_classes = 10      # MNIST有10个类别
    learning_rate = 0.01
    lam = 0.01
    num_epochs = 20
    batch_size = 64

    # 使用 suggest_int 和 suggest_float 创建参数数组
    output_size_list = [trial.suggest_int(f'output_size_{i}', 3000, 5000) for i in range(1)] 
    d_f_list = [trial.suggest_float(f'd_f_{i}', 0.0, 1.0) for i in range(1)]
    alpha_list = [trial.suggest_float(f'alpha_{i}', 0.1, 2.0) for i in range(1)]

    pipeline, deal_train_loader =DFBM(train_loader, output_size_list, d_f_list, alpha_list, device)

    output_size = output_size_list[-1]

    # 定义损失函数和优化器
    model2 = modelset.MLP(output_size, num_classes).to(device)
    criterion2 = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = optim.Adam(model2.parameters(), lr=learning_rate)  # 使用随机梯度下降优化器

    model2.train()
    # 训练模型
    epochs = 30
    for epoch in range(epochs):
        for images, labels in deal_train_loader:
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

    # 设置模型为评估模式
    pipeline.add_model(model2)
    pipeline.eval()

    # 准确率计数
    correct = 0
    total = 0

    # 禁用梯度计算，加速测试过程
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据加载到 GPU
            images = images.view(-1, images_size).to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = pipeline(images)
            
            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            
            # 更新计数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 1.0 * correct / total

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([trial.number, output_size_list, d_f_list, alpha_list, accuracy])


    return 1 - accuracy

# 创建 Optuna study 并优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3000)  # 进行 100 次优化搜索

# 输出最佳参数
print("Best parameters:", study.best_params)
print("Best validation loss:", study.best_value)

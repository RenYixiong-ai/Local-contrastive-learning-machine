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
from train.train import pre_DFBM
from utils import *

import optuna
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(42)

#加载数据集
class_counts = [100]*10
datatype = 'KMNIST'
train_loader = get_dataloader(datatype, batch_size=64, train=True, class_counts=class_counts)
test_loader = get_dataloader(datatype, batch_size=64, train=False)
data_iter = iter(train_loader)
images, labels = next(data_iter)
batch, channel, large, _ = images.shape

csv_file_path = os.path.join(local_path, 'KMNIST_PreDFBM_2_40.csv')

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
    num_epochs = 40
    batch_size = 64

    # 使用 suggest_int 和 suggest_float 创建参数数组
    output_size_list = [trial.suggest_int(f'output_size_{i}', 300, 2000) for i in range(2)] 
    d_f_list = [trial.suggest_float(f'd_f_{i}', 0.0, 1.0) for i in range(2)]
    alpha_list = [trial.suggest_float(f'alpha_{i}', 0.1, 2.0) for i in range(2)]

    pipeline, deal_train_loader = pre_DFBM(train_loader, output_size_list, d_f_list, alpha_list, device, train_MLP=True)

    # 设置模型为评估模式
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

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

from models import modelset as models
from loss.loss import FBMLoss
from loss.loss import test_accuracy

import optuna
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(234)

#加载数据集
class_counts = [100]*10
datatype = 'KMNIST'
train_loader = get_dataloader(datatype, batch_size=64, train=True, class_counts=class_counts)
test_loader = get_dataloader(datatype, batch_size=64, train=False)
data_iter = iter(train_loader)
images, labels = next(data_iter)
batch, channel, large, _ = images.shape

#构造配对方案
import itertools
# 生成 0-9 的所有两两组合（不考虑顺序）
all_combinations = list(itertools.combinations(range(10), 2))
# 函数：检查是否覆盖所有数字 0-9
def covers_all_digits(groups):
    # 获取所有组合中的数字
    digits = set()
    for group in groups:
        digits.update(group)
    # 检查是否覆盖 0-9
    return digits == set(range(10))
# 从所有组合中选择 5 组
valid_groups = []
for groups in itertools.combinations(all_combinations, 5):
    if covers_all_digits(groups):
        valid_groups.append(groups)
use_group = valid_groups[560]

# 使用十个进行分类
use_group = [[i] for i in range(10)]
print(use_group)


csv_file_path = os.path.join(local_path, 'KMNIST_wildFBM_10group.csv')

# 如果文件不存在，则创建文件并写入标题行
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['trial_number', 'output_size_list', 'd_f_list', 'alpha_list', 'accuracy'])

def objective(trial):
    # 定义超参数
    input_size = channel * large**2 # MNIST图像大小是28x28
    num_classes = 10      # MNIST有10个类别
    learning_rate = 0.01
    lam = 0.01
    num_epochs = 80
    batch_size = 64

    # 使用 suggest_int 和 suggest_float 创建参数数组
    NN_outsize_list = [trial.suggest_int(f'output_size_{i}', 300, 2000) for i in range(10)] 
    df_list = [trial.suggest_float(f'd_f_{i}', 0.0, 1.0) for i in range(10)]
    #alpha_list = [trial.suggest_float(f'alpha_{i}', 0.1, 2.0) for i in range(5)]
    alpha_list = [1.0] * 10

    par_model = []
    for NN_outsize, df, alpha, sel_label in zip(NN_outsize_list, df_list, alpha_list, use_group):
        model = modelset.FBMLayer(input_size, NN_outsize).to(device)
        criterion = FBMLoss(NN_outsize, 0.01, df=df, alpha=alpha, if_onehot=True)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        model.train()
        # 训练模型
        for epoch in range(30):
            for images, labels in train_loader:
                # 将图像和标签移动到 GPU 上
                images = images.view(-1, input_size).to(device)  # 展平图像并转移到 GPU
                labels = efface_label(sel_label, labels, 10)
                labels = labels.to(device)  # 标签移动到 GPU
                #labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
                
                # 前向传播
                outputs = model(images)
                #loss = criterion(outputs, labels_one_hot, model.linear.weight)
                loss = criterion(outputs, labels, model.linear.weight)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        par_model.append(model)

    par_model = modelset.ParallelNetworks(par_model).eval()


    # 定义损失函数和优化器
    modelout = modelset.MLP(sum(NN_outsize_list), 10).to(device)
    criterion2 = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = optim.Adam(modelout.parameters(), lr=0.01)  # 使用随机梯度下降优化器

    modelout.train()
    # 训练模型
    max_accury = 0.0
    epochs = 50
    for epoch in range(epochs):
        for images, labels in train_loader:
            # 将图像展平为一维向量，并将标签进行 one-hot 编码
            images = images.view(-1, input_size).to(device)  # 展平图像
            labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)  # 将标签转换为 one-hot 编码

            # 前向传播
            with torch.no_grad():
                deal_images = par_model(images)

            outputs = modelout(deal_images)

            # 计算损失
            loss = criterion2(outputs, labels_one_hot)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            total_model = models.ModelPipeline()
            total_model.add_model(par_model)
            total_model.add_model(modelout)
            max_accury = max(test_accuracy(total_model, test_loader, device), max_accury)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([trial.number, NN_outsize_list, df_list, alpha_list, max_accury])


    return 1 - max_accury

# 创建 Optuna study 并优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3000)  # 进行 100 次优化搜索

# 输出最佳参数
print("Best parameters:", study.best_params)
print("Best validation loss:", study.best_value)

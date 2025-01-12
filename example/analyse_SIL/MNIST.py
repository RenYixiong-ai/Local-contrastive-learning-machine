import os
import sys

local_path = os.getcwd()
# 将项目主目录路径添加到 Python 路径
os.chdir("../../")  # 使用相对路径将工作目录切换到 project 文件夹
project_path = os.path.abspath(os.path.join(local_path, "../../"))
sys.path.append(project_path)   #将模块查找路径切换
local_path = os.path.join(os.getcwd(), "result/analyse_SIL")
os.makedirs(local_path, exist_ok=True)
print(local_path)

import torch
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import utils 
import numpy as np
import random

import torch.optim as optim
import torch.nn.functional as F
from loss.loss import FBMLoss
from models.modelset import FBMLayer

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pickle

# 可视化函数：将数据降维到 2D 并可视化
def visualize_2d(features, labels, savepath=None):
    """
    使用 t-SNE 将高维数据降维到 2D 并可视化。
    :param features: 高维特征数据 (tensor or numpy array)
    :param labels: 标签 (tensor or numpy array)
    """
    # 转换为 numpy
    features_np = features.detach().numpy()
    labels_np = labels.numpy()

    # 使用 t-SNE 降维到 2D
    _, dims =features.shape
    
    if dims > 2:
        #tsne = TSNE(n_components=2, random_state=42, perplexity=40)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_np)
        explained_variance_ratio = pca.explained_variance_ratio_
    else:
        features_2d = features_np
        explained_variance_ratio = None

    # 绘制 2D 散点图
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        mask = labels_np == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            label=f"Class {label}",
            alpha=0.6
        )
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Visualization of High-Dimensional Data")
    plt.legend()
    plt.grid(True)

    # 显示 PCA 信息占比
    if explained_variance_ratio is not None:
        info_text = f"Explained Variance:\nDim 1: {explained_variance_ratio[0]:.2%}\nDim 2: {explained_variance_ratio[1]:.2%}"
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath+'.png')
    plt.close()
    return None

def find_combination_index(selected_array):
    """
    给定一个选取的数组，返回对应的第几种取法的值。

    参数:
    - selected_array: list[int]，选取的数组，元素从 1 到 10，不重复。

    返回:
    - index: int，对应的第几种取法（从 0 开始计数）。
    """
    # 输入校验
    if not all(0 <= x <= 9 for x in selected_array):
        raise ValueError("数组中的元素必须在 0 到 9 之间。")
    if len(set(selected_array)) != len(selected_array):
        raise ValueError("数组中的元素不能重复。")
    
    # 构造二进制状态
    binary_state = [0] * 10
    for num in selected_array:
        binary_state[num] = 1
    
    # 将二进制状态转换为整数（取法索引）
    index = int("".join(map(str, binary_state)), 2)
    return index

utils.set_seed(42)  # 42 是一个示例种子数，您可以根据需求更改
batch_size = 64
selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#selected_classes = [5, 6, 7, 8, 9]
train_loader = utils.get_dataloader("MNIST", batch_size=batch_size, train=True, selected_classes=selected_classes, class_counts=[100]*len(selected_classes))
test_loader = utils.get_dataloader("MNIST", batch_size=batch_size, train=False, selected_classes=selected_classes)

#创建文件位置
local_path = os.path.join(local_path, f"selected_classes{find_combination_index(selected_classes)}")
os.makedirs(local_path, exist_ok=True)

# 获取 DataLoader 中的全部数据
train_features = []
train_labels = []

for batch_data, batch_labels in train_loader:
    train_features.append(batch_data.view(-1, 784))
    train_labels.append(batch_labels)

# 合并所有批次
train_features = torch.cat(train_features, dim=0)
train_labels = torch.cat(train_labels, dim=0)

# 获取 DataLoader 中的全部数据
test_features = []
test_labels = []

for batch_data, batch_labels in train_loader:
    test_features.append(batch_data.view(-1, 784))
    test_labels.append(batch_labels)

# 合并所有批次
test_features = torch.cat(test_features, dim=0)
test_labels = torch.cat(test_labels, dim=0)

# 保存标签到文件
with open(os.path.join(local_path, "train_labels.pkl"), "wb") as file:  # 使用 "wb" 模式打开文件，表示写入二进制数据
    pickle.dump(train_labels, file)
with open(os.path.join(local_path, "test_labels.pkl"), "wb") as file:  # 使用 "wb" 模式打开文件，表示写入二进制数据
    pickle.dump(test_labels, file)
with open(os.path.join(local_path, "train_features.pkl"), "wb") as file:  # 使用 "wb" 模式打开文件，表示写入二进制数据
    pickle.dump(train_features, file)
with open(os.path.join(local_path, "test_features.pkl"), "wb") as file:  
    pickle.dump(test_features, file)

def more_data(a, b):
    # 定义超参数
    input_size = 784
    hidden_dim = 1000
    num_classes = 10      # MNIST有10个类别
    learning_rate = 0.01
    num_epochs = 100


    # 实例化模型、定义损失函数和优化器
    model = FBMLayer(input_size, hidden_dim).to(device)
    criterion = FBMLoss(hidden_dim, 0.01, a, b, losstype="fast_StrongInter")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    # 训练模型
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 将图像和标签移动到 GPU 上
            images = images.view(-1, input_size).to(device)  # 展平图像并转移到 GPU
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
        
        '''分析代码'''
        with torch.no_grad():
            local_data = train_features.view(-1, input_size).to(device)
            analyse_data = model(local_data).cpu()
            
            local_save_path = os.path.join(local_path, f"hidden{hidden_dim}_a{a:.2f}_b{b:.2f}/")
            os.makedirs(local_save_path, exist_ok=True)

            visualize_2d(analyse_data, train_labels.cpu(), os.path.join(local_save_path, f"{epoch}"))   #画出最后的结果
            # 保存数据到文件
            with open(os.path.join(local_save_path, f"train_features_{epoch}.pkl"), "wb") as file:  # 使用 "wb" 模式打开文件，表示写入二进制数据
                pickle.dump(analyse_data, file)


    with torch.no_grad():
        local_data = test_features.view(-1, input_size).to(device)
        analyse_data = model(local_data).cpu()

        visualize_2d(analyse_data, test_labels.cpu(), os.path.join(local_save_path, "test"))   #画出最后的结果
        out_dis = utils.fast_FBDistance(analyse_data, test_labels.cpu())
        # 计算对角项的和
        diagonal_sum = torch.diag(out_dis).sum()

        # 计算非对角项的和
        total_sum = out_dis.sum()
        non_diagonal_sum = total_sum - diagonal_sum
        with open(os.path.join(local_save_path, "distance.txt"), "a") as file:
            file.write("="*20+"\n"+"test"+"\n")
            file.write(str(out_dis) + "\n")  # 写入数据并换行
            F_B = non_diagonal_sum/diagonal_sum
            file.write(f"Bosen={diagonal_sum:.4f}\t Fermi={non_diagonal_sum:.4f}\t Fermi_Bosen={F_B:.4f}\n\n")

'''
for a in np.linspace(0.3, 0.8, 50):
    for b in np.linspace(0.01, 0.5, 50):
        more_data(a, b)
'''

more_data(1.0, 1.0)

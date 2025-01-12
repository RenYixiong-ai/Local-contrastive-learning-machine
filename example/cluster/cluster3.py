import os
import sys

local_path = os.getcwd()
# 将项目主目录路径添加到 Python 路径
os.chdir("../../")  # 使用相对路径将工作目录切换到 project 文件夹
project_path = os.path.abspath(os.path.join(local_path, "../../"))
sys.path.append(project_path)   #将模块查找路径切换
local_path = os.path.join(os.getcwd(), "result/cluster/cluster3")
os.makedirs(local_path, exist_ok=True)
print(local_path)

import torch
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from utils import *
set_seed(42)  # 42 是一个示例种子数，您可以根据需求更改


class_counts = [100]*10
datatype = 'MNIST'
images_size = 1*28*28

train_loader = get_dataloader(datatype, batch_size=64, train=True, class_counts=class_counts)
test_loader = get_dataloader(datatype, batch_size=64, train=False)
"""
train_loader, test_loader = load_small_cifar10(loaad_size=100)
"""
train_features = []
train_labels = []

for batch_data, batch_labels in train_loader:
    train_features.append(batch_data.view(-1, images_size))
    train_labels.append(batch_labels)

# 合并所有批次
train_features = torch.cat(train_features, dim=0)
train_labels = torch.cat(train_labels, dim=0)

# 获取 DataLoader 中的全部数据
test_features = []
test_labels = []

for batch_data, batch_labels in test_loader:
    test_features.append(batch_data.view(-1, images_size))
    test_labels.append(batch_labels)

# 合并所有批次
test_features = torch.cat(test_features, dim=0)
test_labels = torch.cat(test_labels, dim=0)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np

def cluster_accuracy(test_loader, model=None, device=None):
    if model is None:
        deal_test_loader = test_loader
    else:
        deal_test_loader = deal_dataloader(test_loader, model, device, batch_size = 64)

    features, labels = next(iter(deal_test_loader))
    dims = torch.prod(torch.tensor(features.shape[1:]))

    # 获取 DataLoader 中的全部数据
    deal_test_features = []
    deal_test_labels = []

    for batch_data, batch_labels in deal_test_loader:
        deal_test_features.append(batch_data.view(-1, dims))
        deal_test_labels.append(batch_labels)

    # 合并所有批次
    deal_test_features = torch.cat(deal_test_features, dim=0)
    deal_test_labels = torch.cat(deal_test_labels, dim=0)

    data = deal_test_features.numpy()  # 展平
    labels = deal_test_labels.numpy()
    print(labels.shape)

    # 2. 应用KMeans聚类
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(data)

    # 3. 创建混淆矩阵
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int32)
    for cluster in range(n_clusters):
        cluster_labels = labels[cluster_assignments == cluster]
        for label in range(n_clusters):
            confusion_matrix[cluster, label] = np.sum(cluster_labels == label)

    #print("混淆矩阵：")
    #print(confusion_matrix)

    # 4. 使用匈牙利算法找到最佳标签分配
    # 我们需要最大化正确分类，因此将混淆矩阵取负作为成本矩阵
    cost_matrix = -confusion_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 创建标签映射
    cluster_to_label = {}
    for cluster, label in zip(row_ind, col_ind):
        cluster_to_label[cluster] = label

    '''    
    print("\n聚类到标签的映射：")
    for cluster in cluster_to_label:
        print(f"聚类 {cluster} -> 标签 {cluster_to_label[cluster]}")
    '''
    # 5. 使用映射分配标签并计算准确率
    predicted_labels = np.array([cluster_to_label[cluster] for cluster in cluster_assignments])
    accuracy = accuracy_score(labels, predicted_labels)
    #print(f"\n分类准确率: {accuracy * 100:.2f}%")
    return accuracy

from models.modelset import MLP
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def MLP_accuracy(train_loader, test_loader, model=None, device=None):
    if model is not None:
        train_loader = deal_dataloader(train_loader, model, device, batch_size = 64)
        test_loader = deal_dataloader(test_loader, model, device, batch_size = 64)
    learning_rate = 0.01
    num_classes = 10
    # 定义损失函数和优化器
    features, labels = next(iter(test_loader))
    images_size = torch.prod(torch.tensor(features.shape[1:]))
    model0 = MLP(images_size, 10).to(device)
    criterion2 = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = optim.Adam(model0.parameters(), lr=learning_rate)  # 使用随机梯度下降优化器

    model0.train()
    # 训练模型
    epochs = 30
    for epoch in range(epochs):
        for images, labels in train_loader:
            # 将图像展平为一维向量，并将标签进行 one-hot 编码
            images = images.to(device)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)  # 将标签转换为 one-hot 编码

            # 前向传example/cluster/cluster2.py播
            outputs = model0(images)

            # 计算损失
            loss = criterion2(outputs, labels_one_hot)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 设置模型为评估模式
    model0.eval()

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
            outputs = model0(images)
            
            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            
            # 更新计数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 1.0 * correct / total
    return accuracy

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 可视化函数：将数据降维到 2D 并可视化
def visualize_2d(features, labels):
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
    #plt.legend()
    plt.grid(True)

    # 显示 PCA 信息占比
    if explained_variance_ratio is not None:
        info_text = f"Explained Variance:\nDim 1: {explained_variance_ratio[0]:.2%}\nDim 2: {explained_variance_ratio[1]:.2%}"
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))

    plt.show()

import torch.optim as optim
import torch.nn.functional as F
from loss.loss import FBMLoss
from models.modelset import FBMLayer
from models.modelset import FBM_KANLayer

import pickle

# 定义超参数
input_size = images_size
hidden_dim = 1000
num_classes = 10      # MNIST有10个类别
learning_rate = 0.01
num_epochs = 50
batch_size = 64
beta = 2.0
#alpha = 0.006548084
#df = 1.7666495380528981
df = 0.45
alpha=1.0

def layer_train(train_loader, test_loader, df, input_size):
    #记录训练数据
    Boson_list = []
    Fermi_list = []
    mean_norm_list = []
    std_norm_list = []
    cluster_accuracy_list = []
    MLP_accuracy_list = []

    """
    train_loader, test_loader = load_small_cifar10(loaad_size=100)
    """
    train_features = []
    train_labels = []

    for batch_data, batch_labels in train_loader:
        train_features.append(batch_data.view(-1, input_size))
        train_labels.append(batch_labels)

    # 合并所有批次
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # 获取 DataLoader 中的全部数据
    test_features = []
    test_labels = []

    for batch_data, batch_labels in test_loader:
        test_features.append(batch_data.view(-1, input_size))
        test_labels.append(batch_labels)

    # 合并所有批次
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # 实例化模型、定义损失函数和优化器
    model = FBMLayer(input_size, hidden_dim).to(device)
    #model = FBM_KANLayer(input_size, hidden_dim).to(device)
    #criterion = FBMLoss(hidden_dim, 0.01, df, alpha, losstype="fast_StrongInter")
    criterion = FBMLoss(hidden_dim, 0.01, df, alpha, losstype="fast_FermiBose", beta=beta)
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


        """分析数据"""

        # 计算距离
        with torch.no_grad():
            deal_features = model(train_features.to(device))
        FBDis = fast_FBDistance(deal_features, train_labels).numpy()
        Boson = FBDis.trace()
        Fermi = FBDis.sum() - Boson
        Boson_list.append(Boson)
        Fermi_list.append(Fermi)
        print(f"Boson distance={Boson}, Fermi distance={Fermi}")

        # 计算范数
        norms = np.linalg.norm(deal_features.cpu().numpy(), axis=1)
        # 统计范数的均值和标准差
        mean_norm_list.append(np.mean(norms))
        std_norm_list.append(np.std(norms))

        # 计算聚类和全连接的准确率
        cluster_accuracy_list.append(cluster_accuracy(test_loader, model=model, device=device))
        MLP_accuracy_list.append(MLP_accuracy(train_loader, test_loader, model=model, device=device))

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    FerBos = []
    for i in range(num_epochs):
        FerBos.append(Fermi_list[i]/Boson_list[i])
    
    test_loader = deal_dataloader(test_loader, model, device, batch_size = 64)
    train_loader = deal_dataloader(train_loader, model, device, batch_size = 64)
    return Boson_list, Fermi_list, FerBos, train_loader, test_loader, [cluster_accuracy_list, MLP_accuracy_list]

def main(df_list, train_loader, test_loader, name):
    input_size = images_size
    Boson_list, Fermi_list, FerBos_list = [], [], []
    accuracy_list = []
    for df in df_list:
        Boson, Fermi, FerBos, train_loader, test_loader, accuracy = layer_train(train_loader, test_loader, df, input_size)
        Boson_list.append(Boson)
        Fermi_list.append(Fermi)
        FerBos_list.append(FerBos)
        accuracy_list.append(accuracy)
        input_size = 1000

    x = np.arange(1, num_epochs+1)  # 横坐标为 1 到 epoch
    fig, axes = plt.subplots(3, 4, figsize=(20, 16))  # 创建一个包含3行4列子图的图像

    for i in range(3):
        axes[i,0].plot(x, Boson_list[i], label='Boson', marker='o')  # 绘制第一条曲线
        axes[i,0].set_xlabel('Epoch')
        axes[i,0].set_ylabel('Distance')
        axes[i,0].set_title('Boson distance')
        axes[i,0].legend()
        axes[i,0].grid(True)

        axes[i,1].plot(x, Fermi_list[i], label='Fermi', marker='^')  # 绘制第二条曲线
        axes[i,1].set_xlabel('Epoch')
        axes[i,1].set_ylabel('Distance')
        axes[i,1].set_title('Fermi distance')
        axes[i,1].legend()
        axes[i,1].grid(True)

        axes[i,2].plot(x, FerBos_list[i], label='Fermi/Boson')  # 绘制第三条曲线
        axes[i,2].set_xlabel('Epoch')
        axes[i,2].set_ylabel('Fermi/Boson')
        axes[i,2].set_title('ratio')
        axes[i,2].legend()
        axes[i,2].grid(True)
    
        axes[i,3].plot(x, accuracy_list[i][0], label='cluster', marker='^')  # 绘制第二条曲线
        axes[i,3].plot(x, accuracy_list[i][1], label='MLP', marker='o')  # 绘制第二条曲线
        axes[i,3].set_xlabel('Epoch')
        axes[i,3].set_ylabel('Accuracy')
        axes[i,3].set_title('accuracy')
        axes[i,3].legend()
        axes[i,3].grid(True)

    plt.suptitle(datatype)
    plt.savefig(os.path.join(local_path, datatype+name+"_fig.png"))
    plt.close()

    # Save data using pickle
    save_data = {
        "Boson_list": Boson_list,
        "Fermi_list": Fermi_list,
        "accuracy_list": accuracy_list
    }

    with open(os.path.join(local_path, datatype+name+"_data_archive.pkl"), "wb") as file:
        pickle.dump(save_data, file)

for c in np.linspace(0.35, 0.8, 50):
    df_list = [0.30, (0.45+c)/2, c]
    name = f"df0_{df_list[0]:.2f}_df1_{df_list[1]:.2f}_df2_{df_list[2]:.2f}"
    main(df_list, train_loader, test_loader, name)

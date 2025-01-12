import os
import sys

local_path = os.getcwd()
# 将项目主目录路径添加到 Python 路径
os.chdir("../../")  # 使用相对路径将工作目录切换到 project 文件夹
project_path = os.path.abspath(os.path.join(local_path, "../../"))
sys.path.append(project_path)   #将模块查找路径切换
local_path = os.path.join(os.getcwd(), "result/GaussData")
os.makedirs(local_path, exist_ok=True)
print(local_path)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


from loss.loss import FBMLoss
from models.modelset import FBMLayer
from models import modelset
from train.train import train_FBM
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(42)

def generate_sphere_data_with_overlap(
    num_samples_per_class, dimensions, radius_0, radius_1, center_0, center_1
):
    """
    生成两个类别的数据集，每类数据分布在一个高维球体内，并指定球心位置。
    如果生成的点同时属于两类（即满足两个球体的半径约束），则将其标记为标签 2。
    
    :param num_samples_per_class: 每个类别的样本数量
    :param dimensions: 高维空间的维数
    :param radius_0: 类别 0 球体的半径
    :param radius_1: 类别 1 球体的半径
    :param center_0: 类别 0 球体的球心位置 (list or tensor)
    :param center_1: 类别 1 球体的球心位置 (list or tensor)
    :return: 数据集 (features, labels)
    """
    def generate_points(num_samples, dimensions, radius_0, radius_1, center_0, center_1, add_label=1):
        data = []
        labels = []
        center_tensor_0 = torch.tensor(center_0, dtype=torch.float32)
        center_tensor_1 = torch.tensor(center_1, dtype=torch.float32)
        
        while len(data) < num_samples:
            # 从标准高斯分布生成点
            point = torch.randn(dimensions) + center_tensor_0
            
            # 计算点到两个球心的距离
            distance_to_0 = torch.norm(point - center_tensor_0)
            distance_to_1 = torch.norm(point - center_tensor_1)
            
            # 检查点是否在类别 0 和类别 1 的球体内
            in_sphere_0 = distance_to_0 <= radius_0
            in_sphere_1 = distance_to_1 <= radius_1
            
            if in_sphere_0 and in_sphere_1:
                # 如果点同时属于两类，标记为 2
                data.append(point)
                labels.append(2)
            elif in_sphere_0:
                # 如果点仅属于类别 0
                data.append(point)
                labels.append(add_label)


        return torch.stack(data), torch.tensor(labels, dtype=torch.long)
    
    # 生成数据和标签
    features1, labels1 = generate_points(
        num_samples_per_class,  # 生成更多点以处理两类重叠的情况
        dimensions,
        radius_0,
        radius_1,
        center_0,
        center_1,
        add_label = 0
    )
    features2, labels2 = generate_points(
        num_samples_per_class,  # 生成更多点以处理两类重叠的情况
        dimensions,
        radius_1,
        radius_0,
        center_1,
        center_0,
        add_label = 1
    )
    features, labels = torch.cat((features1, features2), dim=0), torch.cat((labels1, labels2), dim=0)

    return features, labels

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_2d(features, labels, save_path=None):
    """
    使用 PCA 或直接可视化 2D 数据，并在图上显示 PCA 信息占比。
    
    :param features: 高维特征数据 (tensor or numpy array)
    :param labels: 标签 (tensor or numpy array)
    :param save_path: 保存图片的路径（如果指定，则保存而不展示）
    """
    # 转换为 numpy
    features_np = features.detach().numpy()
    labels_np = labels.numpy()
    
    # 降维
    _, dims = features.shape
    if dims > 2:
        # 使用 PCA 降维到 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_np)
        explained_variance_ratio = pca.explained_variance_ratio_
    else:
        features_2d = features_np
        explained_variance_ratio = None
    
    # 绘制 2D 散点图
    plt.figure(figsize=(8, 6))
    for label in set(labels_np):
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
    
    # 保存或展示图像
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



def create(dimensions, rate, hidden_dim, df, local_path):
    os.makedirs(local_path, exist_ok=True)
    # 示例：生成每类 500 个样本的高维数据
    num_samples = 500
    radius_0 = np.sqrt(4*dimensions)/2 * rate
    radius_1 = np.sqrt(4*dimensions)/2 * rate
    center_0 = [-1.0] * dimensions  # 类别 0 的球心位置
    center_1 = [1.0] * dimensions  # 类别 1 的球心位置

    # 生成数据
    features, labels = generate_sphere_data_with_overlap(
        num_samples, dimensions, radius_0, radius_1, center_0, center_1
    )
    test_features, test_labels = generate_sphere_data_with_overlap(
        num_samples, dimensions, radius_0, radius_1, center_0, center_1
    )

    # 使用可视化
    visualize_2d(features, labels, save_path=os.path.join(local_path, "train.png"))
    visualize_2d(test_features, test_labels, save_path=os.path.join(local_path, "test.png"))

    # 找到标签为 2 的索引
    idx_2 = (labels == 2)
    num_idx_2 = idx_2.sum().item()
    # 随机生成 0 或 1 的替换标签
    random_labels = torch.randint(0, 2, (num_idx_2,), dtype=torch.long)
    # 替换标签
    labels[idx_2] = random_labels

    # 将 features 和 labels 转换为 TensorDataset
    dataset = TensorDataset(features, labels)

    # 使用 DataLoader 以便进行批量加载
    batch_size = 64  # 设置每个 batch 的大小
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义超参数
    num_classes = 2      # MNIST有10个类别
    learning_rate = 0.01
    num_epochs = 40
    batch_size = 64
    alpha = 1.0

    # 实例化模型、定义损失函数和优化器
    model = FBMLayer(dimensions, hidden_dim).to(device)
    criterion = FBMLoss(hidden_dim, 0.01, df, alpha)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    # 训练模型
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 将图像和标签移动到 GPU 上
            images = images.view(-1, dimensions).to(device)  # 展平图像并转移到 GPU
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

        with torch.no_grad():
            out_features = model(test_features.to(device)).cpu()
            visualize_2d(out_features, test_labels, save_path=os.path.join(local_path, f"epoch{epoch}.png"))


'''
for rate in np.linspace(0.2, 5.0, 20):
    for df in np.linspace(0.01, 0.9, 20):
        for dimensions in range(2, 12, 2):
            for hidden_dim in range(2, 20, 2):        
                input_path = os.path.join(local_path, f"dimension{dimensions}rate{rate:.2f}df{df:.2f}hidden{hidden_dim}")
                create(dimensions, rate, hidden_dim, df, input_path)
'''

from multiprocessing import Pool

def create_task(args):
    """
    包装任务函数，用于 multiprocessing.Pool
    """
    dimensions, rate, hidden_dim, df, local_path = args
    input_path = os.path.join(local_path, f"dimension{dimensions}rate{rate:.2f}df{df:.2f}hidden{hidden_dim}")
    create(dimensions, rate, hidden_dim, df, input_path)

def generate_tasks(local_path):
    """
    生成所有任务参数
    """
    tasks = []
    for rate in np.linspace(0.2, 5.0, 20):
        for df in np.linspace(0.6, 0.9, 20):
            for dimensions in range(2, 12, 2):
                for hidden_dim in range(2, 20, 2):
                    tasks.append((dimensions, rate, hidden_dim, df, local_path))
    return tasks

def main(local_path):
    num_processes = 80  # 进程数

    # 生成任务
    tasks = generate_tasks(local_path)

    # 使用 Pool 分发任务
    with Pool(processes=num_processes) as pool:
        pool.map(create_task, tasks)

main(local_path)



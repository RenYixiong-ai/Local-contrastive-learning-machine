import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt

####################################################
##################### 随机数种子 #####################
####################################################

def set_seed(seed):
    # 设置 Python 内置 random 库的随机种子
    random.seed(seed)
    
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    
    # 如果使用 GPU 进行训练，则需设置以下两个以保证完全的可重复性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多块 GPU
    
    # 设置 CuDNN 后端以确保结果一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


####################################################
##################### 加载数据集 #####################
####################################################

def load_MNIST():
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 下载并加载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 检查数据加载是否成功
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    
    return train_loader, test_loader


def load_small_MNIST(loaad_size=100):
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载完整的 MNIST 数据集
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # 下载并加载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化每个类别的索引列表
    class_indices = {i: [] for i in range(10)}

    # 遍历数据集，记录每个类别的索引
    for idx, (_, label) in enumerate(full_train_dataset):
        if len(class_indices[label]) < loaad_size:  # 每个类别最多选取 loaad_size 个样本
            class_indices[label].append(idx)
        if all(len(class_indices[i]) == loaad_size for i in range(10)):
            break  # 当每个类别都有 loaad_size 个样本时停止

    # 合并所有类别的索引，形成所需的子集
    selected_indices = [idx for indices in class_indices.values() for idx in indices]

    # 创建自定义的训练数据集
    small_train_dataset = Subset(full_train_dataset, selected_indices)
    train_loader = DataLoader(small_train_dataset, batch_size=64, shuffle=True)

    # 检查数据加载是否成功
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Sample labels: {labels}")
    return train_loader, test_loader


def load_FashionMNIST():
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载并加载训练数据集
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 下载并加载测试数据集
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 检查数据加载是否成功
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

    return train_loader, test_loader

def load_cifar10():
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对 RGB 三个通道分别进行归一化
    ])

    # 下载并加载训练数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 下载并加载测试数据集
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 检查数据加载是否成功
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

    return train_loader, test_loader

def load_small_cifar10(loaad_size=100):
    # 定义数据变换
    # 定义数据预处理流程，将图像转换为灰度，并转为张量
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 将RGB转换为单通道灰度图
        transforms.ToTensor(),                         # 转换为Pytorch张量
        transforms.Normalize((0.5,), (0.5,))           # 对单通道灰度图进行归一化
    ])

    # 加载完整的 cifar10 数据集
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    # 下载并加载测试数据集
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化每个类别的索引列表
    class_indices = {i: [] for i in range(10)}

    # 遍历数据集，记录每个类别的索引
    for idx, (_, label) in enumerate(full_train_dataset):
        if len(class_indices[label]) < loaad_size:  # 每个类别最多选取 loaad_size 个样本
            class_indices[label].append(idx)
        if all(len(class_indices[i]) == loaad_size for i in range(10)):
            break  # 当每个类别都有 loaad_size 个样本时停止

    # 合并所有类别的索引，形成所需的子集
    selected_indices = [idx for indices in class_indices.values() for idx in indices]

    # 创建自定义的训练数据集
    small_train_dataset = Subset(full_train_dataset, selected_indices)
    train_loader = DataLoader(small_train_dataset, batch_size=64, shuffle=True)

    # 检查数据加载是否成功
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Sample labels: {labels}")
    return train_loader, test_loader

def get_dataloader(dataset_name, batch_size=64, root='./data', train=True, selected_classes=None, class_counts=None):
    """
    根据 dataset_name 加载不同的数据集，并应用对应的 Normalize 参数。
    
    参数:
        dataset_name (str): 数据集名称，可选 'MNIST', 'FashionMNIST', 'KMNIST', 'CIFAR10', 'CIFAR100'。
        batch_size (int): 数据加载的 batch size。
        root (str): 数据集下载或加载的路径。
        train (bool): 是否加载训练集，False 表示加载测试集。
        selected_classes (list, optional): 需要选择的类别列表，如果为 None，则选择所有类别。
        class_counts (list, optional): 每个类别选择的样本数量，和 selected_classes 一一对应。如果为 None，则选择每个类别的全部样本。
        
    返回:
        DataLoader: 配置好的数据加载器。
    """
    # 定义 Normalize 参数
    normalize_params = {
        'MNIST': ((0.1307,), (0.3081,)),
        'FashionMNIST': ((0.2860,), (0.3530,)),
        'KMNIST': ((0.1904,), (0.3475,)),
        'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'CIFAR100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    }

    # 检查指定的数据集名称是否在支持的列表中
    if dataset_name not in normalize_params:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # 获取对应的 Normalize 参数
    mean, std = normalize_params[dataset_name]

    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 根据 dataset_name 选择数据集
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'KMNIST':
        dataset = datasets.KMNIST(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # 筛选指定类别和数量的样本
    if selected_classes is not None and class_counts is not None:
        # 创建一个空的列表，用于存储符合条件的索引
        selected_indices = []
        class_counts_dict = {cls: count for cls, count in zip(selected_classes, class_counts)}

        # 遍历数据集，筛选出符合条件的样本
        for idx, (_, label) in enumerate(dataset):
            if label in selected_classes and class_counts_dict[label] > 0:
                selected_indices.append(idx)
                class_counts_dict[label] -= 1

            # 如果每个类别的数据量已经达到要求，跳过
            if all(count == 0 for count in class_counts_dict.values()):
                break

        # 创建一个 Subset，返回选择的样本
        dataset = Subset(dataset, selected_indices)

    # 如果没有指定类别，且需要筛选每个类别的数量
    elif selected_classes is None and class_counts is not None:
        # 创建一个空的列表，用于存储符合条件的索引
        selected_indices = []
        class_counts_dict = {i: count for i, count in zip(range(10), class_counts)}  # 默认为10类（MNIST，CIFAR-10，EMNIST等）

        # 遍历数据集，筛选出符合条件的样本
        for idx, (_, label) in enumerate(dataset):
            if class_counts_dict[label] > 0:
                selected_indices.append(idx)
                class_counts_dict[label] -= 1

            # 如果所有类别的数据量已经达到要求，跳过
            if all(count == 0 for count in class_counts_dict.values()):
                break

        # 创建一个 Subset，返回选择的样本
        dataset = Subset(dataset, selected_indices)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

####################################################
##################### 数据集处理 #####################
####################################################

def deal_dataloader(input_loader, model, device, batch_size = 64):
    """
        该函数用于对数据集的处理。当神经网络冻结权重之后，下一层网络的训练输入需要改变。
        该函数通过已经训练好的网络，将数据集处理为下一层输入的形式。
    """

    # 将预处理后的数据存储在一个列表中
    processed_data = []
    labels_data = []

    # 通过预训练网络处理数据
    with torch.no_grad():  # 禁用梯度以加快计算
        for images, labels in input_loader:
            images = images.to(device)  # 确保图像在正确的设备上
            
            # 展平图像并传入网络中
            features = model(images)
            
            # 将输出特征保存到列表中
            processed_data.append(features.cpu())  # 转到 CPU 上，以便后续处理
            labels_data.append(labels.cpu())

    # 将 processed_data 转换为一个张量，按需进一步处理
    processed_data = torch.cat(processed_data, dim=0)
    labels_data = torch.cat(labels_data, dim=0)
    #print(f"Processed data shape: {processed_data.shape}")

    # 创建 TensorDataset，将处理后的特征和标签配对
    output_dataset = TensorDataset(processed_data, labels_data)

    # 创建 DataLoader，用于迭代访问数据
    output_data_loader = DataLoader(output_dataset, batch_size=batch_size, shuffle=True)

    return output_data_loader

####################################################
##################### 对标签处理 #####################
####################################################

def efface_label(selected_labels, labels, num_classes):
    """
    根据指定的标签列表对输入标签进行处理，将指定的标签转换为 one-hot 编码，
    其余未指定的标签对应的 one-hot 编码保持全零。

    参数:
        selected_labels (list or set): 一个包含指定标签的列表或集合。只有这些标签会被转换为 one-hot 编码，其余标签的 one-hot 编码保持全零。
        labels (torch.Tensor): 输入的标签张量，形状为 (batch_size,)，每个值是一个整数，表示标签类别。
        num_classes (int): 标签的类别总数，用于生成 one-hot 编码的长度。

    返回:
        torch.Tensor: 处理后的 one-hot 编码张量，形状为 (batch_size, num_classes)。其中，
                      - 如果标签在 `selected_labels` 中，则生成对应的 one-hot 编码。
                      - 如果标签不在 `selected_labels` 中，则对应的行全为 0。
    """
    # 创建一个全零的 one-hot 编码张量，形状为 (batch_size, num_classes)
    one_hot_labels = torch.zeros((labels.size(0), num_classes))

    # 遍历每个输入标签
    for i, label in enumerate(labels):
        # 如果标签在 selected_labels 中，将其转换为 one-hot 编码
        if label.item() in selected_labels:
            one_hot_labels[i] = F.one_hot(label, num_classes=num_classes).float()

    return one_hot_labels

####################################################
################# 计算Feri-Bose距离 #################
####################################################
def fast_FBDistance(features, labels):
    """
    计算不同类别之间的平均距离矩阵（Fermi-Bose 距离矩阵）。
    输入:
        features: 张量，形状为 (N, D)，每一行是一个样本的特征向量。
        labels: 张量，形状为 (N,)，表示样本的类别标签。
    输出:
        labels_matrix: 张量，形状为 (C, C)，表示类别间的平均距离矩阵。
            - 对角线 (i, i): 表示类别 i 内样本之间的平均距离。
            - 非对角线 (i, j): 表示类别 i 和类别 j 的样本间的平均距离。
    """

    # 获取所有唯一的类别标签
    labels_list = torch.unique(labels)
    labels_count = len(labels_list)  # 唯一类别的数量

    #获取维度
    _, dim = features.shape

    # 构建每个类别对应的样本特征集合
    # features_list[i] 是类别 i 的所有样本特征的列表
    features_list = [[] for _ in range(labels_count)]
    for i in range(labels_count):
        idx_label = (labels == i)  # 找到标签等于 i 的样本索引
        features_list[i] = features[idx_label]  # 取出属于类别 i 的样本特征

    # 初始化类别间距离矩阵 labels_matrix
    # 形状为 (C, C)，初始值为 0
    labels_matrix = torch.zeros(labels_count, labels_count)

    # 计算类别间的平均距离
    for i in range(labels_count):
        for j in range(i, labels_count):  # 只计算上三角部分 (i <= j)
            # 计算类别 i 和类别 j 的样本之间的两两差值
            # features_list[i].unsqueeze(0): 类别 i 的样本扩展维度以进行广播
            # features_list[j].unsqueeze(1): 类别 j 的样本扩展维度以进行广播
            sample_diff = features_list[i].unsqueeze(0) - features_list[j].unsqueeze(1)

            # 计算两两样本的欧几里得距离矩阵 D_matrix
            # D_matrix 的形状为 (len(features_list[i]), len(features_list[j]))
            D_matrix = torch.norm(sample_diff, dim=2)/dim

            # 计算类别 i 和类别 j 的平均距离
            # 总距离除以样本对的数量
            labels_matrix[i, j] = torch.sum(D_matrix) / (len(features_list[i]) * len(features_list[j]))

    return labels_matrix  # 返回类别间的平均距离矩阵

if __name__ == "__main__":
    set_seed(42)

    # 使用示例
    # 假设我们选择 CIFAR-10 数据集，并选择类别 0 (飞机), 1 (汽车) 各 100 个样本
    selected_classes = [0, 1]
    class_counts = [100, 100]

    # 假设我们选择 CIFAR-10 数据集，并且不指定类别，但是限制每种目标的数目
    #class_counts = [100] * 10

    train_loader = get_dataloader('CIFAR10', batch_size=64, train=True, selected_classes=selected_classes, class_counts=class_counts)
    test_loader = get_dataloader('CIFAR10', batch_size=64, train=False)

    # 检查数据加载是否成功
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
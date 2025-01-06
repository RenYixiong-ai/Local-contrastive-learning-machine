from loss.loss import FBMLoss

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import models 
from utils import *

def train_FBM(input_size, output_size, lam, d_f, alpha, learning_rate, train_loader, num_epochs, device, losstype="fast_FermiBose"):
    """
        输入数据集、网络基本信息、FBM训练基本参数，返回训练好的FBM层
    """
    model = models.FBMLayer(input_size, output_size).to(device)
    criterion = FBMLoss(output_size, lam, d_f, alpha, losstype=losstype)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 将图像和标签移动到 GPU 上
            images = images.view(-1, input_size).to(device)  # 展平图像并转移到 GPU
            labels = labels.to(device)  # 标签移动到 GPU
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels, model.linear.weight)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

def train_MLP(input_size, train_loader, model, criterion, optimizer, num_epochs, device):
    model.train()
    # 训练模型
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 将图像和标签移动到 GPU 上
            images = images.view(-1, input_size).to(device)  # 展平图像并转移到 GPU
            labels = labels.to(device)  # 标签移动到 GPU
            
            # 前向传播
            outputs = model(images)
            #loss = criterion(outputs, labels_one_hot, model.linear.weight)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return None

def DFBM(train_loader, hidden_list, df_list, alpha_list, device, lr=0.01, epoch=30, lam=0.01, losstype="fast_FermiBose"):
    '''
        逐层训练深度FBM，训练完一层之后固定权重再训练下一层权重
    '''
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    batch, channel, large, _ = images.shape

    # 定义超参数 
    input_size = channel * large**2 # MNIST图像大小是28x28
    learning_rate = lr
    lam = lam
    num_epochs = epoch

    pipeline = models.ModelPipeline()
    for output_size, d_f, alpha in zip(hidden_list, df_list, alpha_list):
        model = train_FBM(input_size = input_size, 
                        output_size = output_size, 
                        lam = lam, 
                        d_f = d_f, 
                        alpha = alpha,
                        learning_rate = learning_rate, 
                        train_loader = train_loader, 
                        num_epochs = num_epochs, 
                        device = device,
                        losstype=losstype)

        train_loader = deal_dataloader(train_loader, model, device)
        pipeline.add_model(model)

        input_size = output_size

    return pipeline, train_loader



def pre_DFBM(train_loader, hidden_list, df_list, alpha_list, device, train_MLP=False, num_classes=10, lr=0.01, num_epoch=30, lam=0.01, losstype="fast_FermiBose"):
    '''
    联合训练深度FBM网络
    整体进行推理，对每一层网络单独设计并训练。
    
    参数:
        train_loader (DataLoader): PyTorch数据加载器，用于提供训练数据。
        hidden_list (list[int]): 每层FBM网络的输出特征维度列表，表示每一层的隐藏单元数。
        df_list (list[float]): 每层FBM网络的 d_f 参数列表，控制各层网络的正则化参数。
        alpha_list (list[float]): 每层FBM网络的 alpha 参数列表，控制各层的损失权重。
        device (torch.device): 设备对象，用于指定训练在CPU还是GPU上进行。
        train_MLP (bool, optional): 是否在最后一层添加一个MLP分类器，默认为 False。
        num_classes (int, optional): 最终分类的类别数，仅在 `train_MLP=True` 时有效，默认为 10。
        lr (float, optional): 优化器的学习率，默认为 0.01。
        num_epoch (int, optional): 训练的轮数，默认为 30。
        lam (float, optional): 损失函数的正则化参数，默认为 0.01。
        losstype (str): 选择使用的损失函数类型。
    
    返回:
        tuple:
            - pipeline (ModelPipeline): 包含所有FBM层的模型管道，支持后续推理使用。
            - train_loader (DataLoader): 修改后的训练数据加载器，基于管道模型的输出进行处理。
    '''

    # 获取数据集基本信息
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(images.shape)
    batch, channel, pixel_size, _ = images.shape
    
    # 逐步构建深度FBM
    model_list = []
    criterion_list = []
    optimizer_list = []

    input_size = channel*pixel_size**2
    for output_size, d_f, alpha in zip(hidden_list, df_list, alpha_list):
        model = models.FBMLayer(input_size, output_size).to(device)     # 网络
        criterion = FBMLoss(output_size, lam, d_f, alpha, losstype=losstype)               # 损失
        optimizer = optim.Adam(model.parameters(), lr=lr)               # 优化器

        model_list.append(model)
        criterion_list.append(criterion)
        optimizer_list.append(optimizer)

        input_size = output_size
    
    # 构建最后一层全连接网络
    if train_MLP:
        MLP_model = models.MLP(input_size, num_classes).to(device)
        MLP_criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
        MLP_optimizer = optim.Adam(MLP_model.parameters(), lr=lr)

    # 整体训练模型
    input_size = channel*pixel_size**2
    for epoch in range(num_epoch):
        for images, labels in train_loader:
            # 将图像和标签移动到 GPU 上
            outputs = images.view(-1, input_size).to(device)  # 展平图像并转移到 GPU
            labels = labels.to(device)  # 标签移动到 GPU

            loss_list = []
            # 逐层推理，并且对该层进行训练
            for model, loss, optimizer in zip(model_list, criterion_list, optimizer_list):
                # 前向传播
                inputs = outputs.detach()   #切断梯度图，只计算这一层的梯度
                outputs = model(inputs)
                loss = criterion(outputs, labels, model.linear.weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(f'{loss.item():.4f}')

            if not train_MLP: continue
            inputs = outputs.detach()
            outputs = MLP_model(inputs)
            loss = MLP_criterion(outputs, labels)
            MLP_optimizer.zero_grad()
            loss.backward()
            MLP_optimizer.step()
            loss_list.append(f'{loss.item():.4f}')

                
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss_list}')

    # 构建管线
    pipeline = models.ModelPipeline()
    for model in model_list:
        pipeline.add_model(model)
    if train_MLP: pipeline.add_model(MLP_model)

    # 数据集的修改
    train_loader = deal_dataloader(train_loader, pipeline, device)

    return pipeline, train_loader





if __name__ == "__main__":
    pass



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import modelset
from train.train import train_FBM
from utils import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(42)

class_counts = [100]*10
datatype = 'MNIST'

train_loader = get_dataloader(datatype, batch_size=64, train=True, class_counts=class_counts)
test_loader = get_dataloader(datatype, batch_size=64, train=False)

data_iter = iter(train_loader)
images, labels = next(data_iter)
batch, channel, large, _ = images.shape


# 定义超参数
input_size = channel * large**2 # MNIST图像大小是28x28
hidden_dim =1000
num_classes = 10      # MNIST有10个类别
learning_rate = 0.01
lam = 0.01
d_f = 0.43
alpha = 7.0
num_epochs = 5
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
epochs = 20
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
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


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
print(f'Accuracy on the test dataset: {accuracy:.2f}%')

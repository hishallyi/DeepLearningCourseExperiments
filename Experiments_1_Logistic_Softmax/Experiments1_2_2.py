import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 人工构造数据集
np.random.seed(0)
# 生成两个类别的样本，每个样本有两个特征
features = np.random.randn(200, 2).astype(np.float32)
# 生成随机的标签，0或1
labels = np.random.randint(0, 2, (200,)).astype(np.float32)

# 将数据转换为Tensor
features = torch.from_numpy(features)
labels = torch.from_numpy(labels)


# 构建logistic回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


input_size = 2
model = LogisticRegression(input_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# 可视化损失函数曲线
plt.plot(losses)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Logistic Training Loss : Implement from torch.nn')
plt.show()

# 计算训练集上的准确率
with torch.no_grad():
    predicted = model(features)
    predicted_labels = (predicted > 0.5).float()
    accuracy = (predicted_labels == labels).sum().item() / labels.size(0)

print("训练集上的准确率: {:.2f}%".format(accuracy))

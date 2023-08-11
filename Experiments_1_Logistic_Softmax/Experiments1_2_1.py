import numpy as np
import torch
import matplotlib.pyplot as plt

# 设置随机种子以保证可复现性
np.random.seed(0)

# 生成数据集
num_samples = 1000
num_features = 2

# 生成特征矩阵
X = np.random.randn(num_samples, num_features).astype(np.float32)

# 生成标签矩阵
true_weights = np.array([2, -3]).astype(np.float32)
true_bias = 1
logits = np.dot(X, true_weights) + true_bias
probabilities = 1 / (1 + np.exp(-logits))
y = np.random.binomial(1, probabilities, size=num_samples).astype(np.float32)

# 将数据转换为PyTorch的张量
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)


# 定义模型类
class LogisticRegression:
    def __init__(self, num_features):
        self.weights = torch.zeros(num_features, 1, dtype=torch.float32, requires_grad = True)
        self.bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(self, X):
        linear = torch.matmul(X, self.weights) + self.bias
        return torch.sigmoid(linear)


# 定义二元交叉熵损失函数
def binary_cross_entropy(y_pred, y_true):
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

# 定义训练函数
def train(model, X, y, num_epochs, learning_rate):
    losses = []
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model.forward(X)

        # 计算损失
        loss = binary_cross_entropy(y_pred, y)

        # 反向传播
        loss.backward()

        # 参数更新
        model.weights.data -= learning_rate * model.weights.grad
        model.bias.data -= learning_rate * model.bias.grad

        # 清除梯度
        model.weights.grad.zero_()
        model.bias.grad.zero_()

        if (epoch + 1) % 100 == 0:
            losses.append(loss.item())
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

    epoches = [(i + 1) * 100 for i in range(10)]
    plt.figure(dpi=200)
    plt.plot(epoches, losses, c='r')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('Logistic Training Loss : Implement from scratch')
    plt.show()


# 初始化模型
model = LogisticRegression(num_features)

# 设置训练超参数
num_epochs = 1000
learning_rate = 0.1

# 训练模型
train(model, X_tensor, y_tensor, num_epochs, learning_rate)

# 预测类别
predictions = (model.forward(X_tensor) > 0.5).float()

# 计算训练集的准确率
accuracy = torch.mean((predictions == y_tensor).float())
print(f'Training Accuracy: {accuracy.item()}')

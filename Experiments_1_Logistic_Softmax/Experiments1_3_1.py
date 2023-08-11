import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 加载FashionMNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, download=False, transform=transform)

# 将数据集分为训练集和测试集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


# 定义softmax回归模型
class SoftmaxRegression:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
        self.W = np.random.randn(self.num_features, self.num_classes) * 0.01
        self.b = np.zeros(self.num_classes)
        self.losses = []

    # 定义softmax函数，在前向传播时使用
    def softmax(self, X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_probs) / m
        return loss

    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # 前向传播
            scores = np.dot(X, self
                            .W) + self.b
            probs = self.softmax(scores)

            # 计算损失
            loss = self.cross_entropy_loss(probs, y)

            # 反向传播
            dsoftmax = probs.copy()
            dsoftmax[range(X.shape[0]), y] -= 1
            dsoftmax /= X.shape[0]

            dW = np.dot(X.T, dsoftmax)
            db = np.sum(dsoftmax, axis=0)

            # 更新参数
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if (epoch + 1) % 100 == 0:
                self.losses.append(loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

    def predict(self, X):
        scores = np.dot(X, self.W) + self.b
        return np.argmax(scores, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


# 初始化模型和超参数
num_features = 28 * 28
num_classes = 10
model = SoftmaxRegression(num_features, num_classes)
num_epochs = 1000
learning_rate = 0.01

# 在训练集上训练模型
X_train = train_dataset.data.numpy().reshape(-1, num_features) / 255.0
y_train = train_dataset.targets.numpy()
model.train(X_train, y_train, num_epochs, learning_rate)

# 在训练集和测试集上进行评估
X_test = test_dataset.data.numpy().reshape(-1, num_features) / 255.0
y_test = test_dataset.targets.numpy()
train_accuracy = model.evaluate(X_train, y_train)
test_accuracy = model.evaluate(X_test, y_test)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

epoches = [i+1 for i in range(10)]
plt.figure(dpi=200)
plt.plot(epoches, model.losses, c='b')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Softmax : Implement from scratch')
# plt.savefig('Experiments1_3_1.png')
plt.show()
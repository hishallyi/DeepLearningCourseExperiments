import torch

# 创建随机数矩阵P和Q
P = torch.normal(mean=0.0, std=0.01, size=(3, 2))
Q = torch.normal(mean=0.0, std=0.01, size=(4, 2))

# 对矩阵Q进行形状变换得到Q的转置QT
QT = Q.t()

# 求矩阵相乘P和Q的转置QT
result = torch.matmul(P, QT)

print("矩阵P:")
print(P)
print("矩阵Q:")
print(Q)
print("矩阵QT:")
print(QT)
print("矩阵乘积P * QT:")
print(result)
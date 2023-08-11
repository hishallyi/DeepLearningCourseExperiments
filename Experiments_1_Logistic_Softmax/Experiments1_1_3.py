import torch

x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2

# 中断梯度的追踪
with torch.no_grad():
    y2 = x ** 3

y3 = y1 + y2

# 计算 y3 对 x 的梯度
y3.backward()

print(x.grad)

import torch

# 初始化矩阵M和N
M = torch.Tensor([[1, 2, 3]])
N = torch.tensor([[4], [5]])

# 方式1：使用减法操作符“-”
result1 = M - N

# 方式2：使用torch.sub函数
result2 = torch.sub(M, N)

# 方式3：使用torch.subtract函数
result3 = torch.subtract(M, N)

# 打印结果
print("方式1的结果：", result1)
print("方式2的结果：", result2)
print("方式3的结果：", result3)
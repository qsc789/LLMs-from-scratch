import torch
import torch.nn as nn
torch.manual_seed(123)

# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2, 5)# 初始化随机数

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())# 定义网络结构
out = layer(batch_example)# 前向传播
print(out)
mean = out.mean(dim=-1, keepdim=True)# 计算均值
var = out.var(dim=-1, keepdim=True)# 计算方差

print("Mean:\n", mean)
print("Variance:\n", var)
out_norm = (out - mean) / torch.sqrt(var)# 归一化
print("Normalized layer outputs:\n", out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)# 计算归一化后的均值
var = out_norm.var(dim=-1, keepdim=True)# 计算归一化后端方差
print("Mean:\n", mean)
print("Variance:\n", var)
torch.set_printoptions(sci_mode=False)# 取消科学计数法
print("Mean:\n", mean)
print("Variance:\n", var)
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))# 初始化缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))# 初始化偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)# 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)# 计算方差，不使用偏置项
        norm_x = (x - mean) / torch.sqrt(var + self.eps)# 归一化
        return self.scale * norm_x + self.shift
ln = LayerNorm(emb_dim=5)# 定义层归一化层
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)
ln = LayerNorm(emb_dim=5)# 定义层归一化层
out_ln = ln(batch_example)# 前向传播
mean = out_ln.mean(dim=-1, keepdim=True)# 归一化后均值
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)# 归一化后方差

print("Mean:\n", mean)
print("Variance:\n", var)
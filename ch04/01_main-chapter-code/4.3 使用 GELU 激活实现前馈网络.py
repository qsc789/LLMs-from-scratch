import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 设置环境变量以允许重复加载OpenMP运行时
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):# forward函数旨在计算GELU激活函数的输出
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


gelu, relu = GELU(), nn.ReLU()# 实例化GELU和ReLU激活函数

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)# 计算GELU和ReLU激活函数的输出

plt.figure(figsize=(8, 3))# 绘制两个函数的曲线图
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )# 定义前馈网络的结构，包括两个全连接层，一个GELU激活函数

    def forward(self, x):
        return self.layers(x)
print(GPT_CONFIG_124M["emb_dim"])# 打印GPT-124M模型的embedding维度
ffn = FeedForward(GPT_CONFIG_124M)# 实例化前馈网络
# input shape: [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768) # 随机输入
out = ffn(x)# 前馈网络的输出
print(out.shape)
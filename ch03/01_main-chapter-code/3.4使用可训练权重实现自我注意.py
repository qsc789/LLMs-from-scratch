import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2
torch.manual_seed(123)# 设置随机数种子
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)# 初始化3个权重矩阵
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
keys = inputs @ W_key# 计算所有输入向量与第二个输入向量的注意力权重
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)# 计算第二个输入向量与所有输入向量的注意力权重
print(attn_score_22)
attn_scores_2 = query_2 @ keys.T# 同上
print(attn_scores_2)
# 我们使用之前使用的 softmax 函数计算注意力权重（总和为 1 的标准化注意力分数）
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
# 计算输入查询向量 2 的上下文向量
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# 3.4.2 实现一个紧凑的 SelfAttention 类
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key# 计算3个权重矩阵
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1# 计算注意力权重，dim=-1表示对最后一维进行softmax
        )

        context_vec = attn_weights @ values# 计算上下文向量
        return context_vec


torch.manual_seed(123)# 随机数种子
sa_v1 = SelfAttention_v1(d_in, d_out)# 初始化SelfAttention——v1类
print(sa_v1(inputs))


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)# bias=qkv_bias表示是否使用偏置
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))# SelfAttention_v1 和 SelfAttention_v2 给出不同的输出，因为它们对权重矩阵使用不同的初始权重
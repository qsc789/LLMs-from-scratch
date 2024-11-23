import torch.nn as nn
import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)# 3个权重矩阵
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # 注意力分数为queries与keys的乘积
        attn_scores.masked_fill_(  # 掩盖未来注意力权重
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(# 用softmax归一化注意力权重
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # 应用dropout

        context_vec = attn_weights @ values# 计算上下文向量
        return context_vec
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(# 定义多个注意力头
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# 嵌入维度为 4，因为我们d_out=2 作为键、查询和值向量以及上下文向量的嵌入维度。由于我们有 2 个注意力头
# 因此我们得到了输出嵌入维度 2*2=4
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)# batch是两个输入序列
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(# 定义多头注意力
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)# 计算上下文向量

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# 3.6.2 使用权重拆分实现多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"# 输出维度必须能被num_heads整除

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 减少输出维度以匹配期望输出维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 线性层组合头输出
        self.dropout = nn.Dropout(dropout)# 应用dropout
        self.register_buffer(# 注册缓冲区
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)# 掩盖未来注意力权重
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 我们隐式的通过增加num_haeds维度来分割矩阵
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算点积注意力
        attn_scores = queries @ keys.transpose(2, 3)  # 注意力分数为queries与keys的乘积

        # 原始掩码截断为token数并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 用mask填充注意力分散
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)# softmax归一化注意力权重
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)# 组合头输出
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)# 定义多头注意力

context_vecs = mha(batch)# 计算上下文向量

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print(a @ a.transpose(2, 3))
# 一种更紧凑的方法
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T# 第一个头的注意力分散
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T# 第二个头的注意力分散
print("\nSecond head:\n", second_res)
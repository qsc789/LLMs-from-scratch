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
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2
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
# Reuse the query and key weight matrices of the SelfAttention_v2 object from the previous section for convenience
# 之前的分数和权重
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
# 掩盖未来注意力权重的最简单方法是通过 PyTorch 的 tril 函数创建一个掩码
# 其中主对角线下方的元素（包括对角线本身）设置为 1，主对角线上方的元素设置为 0：
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))# 掩码矩阵
print(mask_simple)
# 然后用这个掩码将注意力权重相乘，将对角线以上的注意力分数归零
masked_simple = attn_weights*mask_simple
print(masked_simple)
row_sums = masked_simple.sum(dim=-1, keepdim=True)# 求每行的和，dim=-1表示按行求和
masked_simple_norm = masked_simple / row_sums# 算权重
print(masked_simple_norm)
# 用triu函数创建掩码矩阵
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)# 将对角线上方元素设为负无穷
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)# 计算注意力权重
print(attn_weights)

# 3.5.2 使用 dropout 屏蔽额外的注意力权重
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
example = torch.ones(6, 6) # create a matrix of ones
print(dropout(example))
torch.manual_seed(123)
print(dropout(attn_weights))# dropout屏蔽额外的注意力权重

# 3.5.3 实现紧凑的因果自注意力类
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3
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

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)# 0.0 dropout rate

context_vecs = ca(batch)# 计算上下文向量

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
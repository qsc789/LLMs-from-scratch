import torch
import torch.nn as nn
import tiktoken
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
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
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):# forward函数旨在计算GELU激活函数的输出
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


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
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(# 注意力层
            d_in=cfg["emb_dim"],# 输入维度
            d_out=cfg["emb_dim"],# 输出维度
            context_length=cfg["context_length"],# 注意力上下文长度
            num_heads=cfg["n_heads"], # 多头注意力头数
            dropout=cfg["drop_rate"],# dropout率
            qkv_bias=cfg["qkv_bias"])# 是否使用qkv偏置
        self.ff = FeedForward(cfg)# 前馈网络层
        self.norm1 = LayerNorm(cfg["emb_dim"])# 归一化层1
        self.norm2 = LayerNorm(cfg["emb_dim"])# 归一化层2
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])# 使用dropout

    def forward(self, x):
        # 注意力快的短连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加原始输入

        # 前馈网络的短连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加原始输入

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])# 词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])# 位置潜入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])# 嵌入层dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])# 多层Transformer块

        self.final_norm = LayerNorm(cfg["emb_dim"])# 最终归一化层
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False# 输出层
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape# 获取输入的batch_size
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
tokenizer = tiktoken.get_encoding("gpt2")# 加载tokenizer
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))# 编码文本
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)# 堆叠batch
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
total_params = sum(p.numel() for p in model.parameters())# 计算模型参数数量
print(f"Total number of parameters: {total_params:,}")# 输出模型参数数量
print("Token embedding layer shape:", model.tok_emb.weight.shape)# 输出词潜入层shape
print("Output layer shape:", model.out_head.weight.shape)# 输出输出层shape
total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())# 计算模型参数数量，考虑权重共享
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")# 输出模型参数数量
# 计算模型大小，单位为MB
total_size_bytes = total_params * 4
# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)# 输出模型大小
print(f"Total size of the model: {total_size_mb:.2f} MB")
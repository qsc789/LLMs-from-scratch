import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # 如果当前上下文超过支持的上下文大小，则裁剪当前上下文
        # 例如，如果语言模型仅支持5个token，而上下文大小为10，则当前上下文将被裁剪为最后5个token
        # 然后只使用最后5个token作为上下文
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)# 得到模型预测结果

        # 只取最后一个时间步的预测结果
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # 用softmax得到概率分布
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # 得到概率最大的词的idx
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # 添加到当前序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
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
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))# 位置嵌入
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)# 应用dropout
        x = self.trf_blocks(x)# 应用多层Transformer块
        x = self.final_norm(x)# 应用最终归一化层
        logits = self.out_head(x)# 输出层
        return logits
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})# 编码文本
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 添加batch维度
    return encoded_tensor
model = GPTModel(GPT_CONFIG_124M)# 加载模型

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 去掉batch维度
    return tokenizer.decode(flat.tolist())
model.to("cpu")# 加载模型到CPU
model.eval()# 评估模式

tokenizer = tiktoken.get_encoding("gpt2")# 定义模型

token_ids = generate_text_simple(# 生成文本
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),# 输入文本
    max_new_tokens=25,# 生成最大token数
    context_size=GPT_CONFIG_124M["context_length"]# 注意力上下文长度
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))# 输出文本

# 5.3.1 温度缩放
vocab = {# 定义词典
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}# 反向词典

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(# 下一个token的logits
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)# 计算softmax概率分布
next_token_id = torch.argmax(probas).item()# 得到概率最大的词的idx

# 下一个生成的token为
print(inverse_vocab[next_token_id])
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()# 得到随机采样的词的idx
print(inverse_vocab[next_token_id])
def print_sampled_tokens(probas):# 打印采样的词
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]# 得到1000个随机采样的词的idx
    sampled_ids = torch.bincount(torch.tensor(sample))# 统计词频
    for i, freq in enumerate(sampled_ids):# 打印词频
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)# 打印采样的词
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature# 缩放logits
    return torch.softmax(scaled_logits, dim=0)# 计算softmax概率分布

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence原始，更高的置信度，更低的置信度

# 计算缩放概率
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
# Plotting
x = torch.arange(len(vocab))# 定义x轴
bar_width = 0.15# 定义柱状图宽度

fig, ax = plt.subplots(figsize=(5, 3))# 定义画布
for i, T in enumerate(temperatures):# 画柱状图
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')# 画柱状图

ax.set_ylabel('Probability')# 设置y轴标签
ax.set_xticks(x)# 设置x轴刻度
ax.set_xticklabels(vocab.keys(), rotation=90)# 设置x轴标签
ax.legend()# 显示图例

plt.tight_layout()# 自动调整子图间距
plt.savefig("temperature-plot.pdf")# 保存图片
plt.show()
print_sampled_tokens(scaled_probas[1])# 打印采样的词
print_sampled_tokens(scaled_probas[2])# 打印采样的词

# 5.3.2 Top-k 采样
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)# 得到top_k的logits和位置

print("Top logits:", top_logits)# 打印top_k的logits
print("Top positions:", top_pos)# 打印top_k的位置
new_logits = torch.where(# 得到新的logits
    condition=next_token_logits < top_logits[-1],# 条件：当前logits小于top_k的logits的最大值
    input=torch.tensor(float("-inf")),# 输入：负无穷
    other=next_token_logits# 其他：当前logits
)

print(new_logits)
topk_probas = torch.softmax(new_logits, dim=0)# 计算新的softmax概率分布
print(topk_probas)

# 5.3.3 修改文本生成功能
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]# 裁剪当前上下文
        with torch.no_grad():# 不计算梯度
            logits = model(idx_cond)# 得到模型预测结果
        logits = logits[:, -1, :]# 只取最后一个时间步的预测结果

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # 只保留top_k个值
            top_logits, _ = torch.topk(logits, top_k)# 得到top_k的logits和位置
            min_val = top_logits[:, -1]# 得到top_k的logits的最大值
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)# 得到新的logits

        # New: Apply temperature scaling
        if temperature > 0.0:# 只有当temperature>0时才进行temperature scaling
            logits = logits / temperature# 缩放logits

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)# 计算softmax概率分布

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # 停止生成，如果指定了eos_id
            break

        # 和之前一样，把采样的idx添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
torch.manual_seed(123)

token_ids = generate(# 生成文本
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
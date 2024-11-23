# 5.1.1 使用 GPT 生成文本
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
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
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))# 位置嵌入
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)# 应用dropout
        x = self.trf_blocks(x)# 应用多层Transformer块
        x = self.final_norm(x)# 应用最终归一化层
        logits = self.out_head(x)# 输出层
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)# 定义模型
model.eval();  # 进入评估模式
import tiktoken
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

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})# 编码文本
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 添加batch维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 去掉batch维度
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")# 定义tokenizer

token_ids = generate_text_simple(# 调用生成文本函数
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))# 打印输出文本
# 5.1.2 计算文本生成损失：交叉熵和困惑度
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]
with torch.no_grad():
    logits = model(inputs)# 得到模型预测结果

probas = torch.softmax(logits, dim=-1) # 每个词的概率分布
print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)# 得到概率最大的词的idx
print("Token IDs:\n", token_ids)# 打印概率最大词的idx
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")# 打印目标文本
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")# 打印输出文本
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]# 计算目标词的概率分布
print("Text 1:", target_probas_1)# 打印目标词的概率分布

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]# 计算目标词的概率分布
print("Text 2:", target_probas_2)# 打印目标词概率分布
# 计算所有词的对数概率
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# 计算每个词的平均对数概率
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
neg_avg_log_probas = avg_log_probas * -1# 计算负对数似然损失
print(neg_avg_log_probas)
# Logits have shape (batch_size, num_tokens, vocab_size)
print("Logits shape:", logits.shape)
# Targets have shape (batch_size, num_tokens)
print("Targets shape:", targets.shape)
logits_flat = logits.flatten(0, 1)# 展平logits
targets_flat = targets.flatten()# 展平targets

print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)# 计算交叉损失
print(loss)
perplexity = torch.exp(loss)# 计算困惑度
print(perplexity)



# 5.1.3 计算训练集和验证集损失
import os
import urllib.request

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
# First 100 characters
print(text_data[:99])
# Last 100 characters
print(text_data[-99:])
total_characters = len(text_data)# 总字符数
total_tokens = len(tokenizer.encode(text_data))# 总token数

print("Characters:", total_characters)# 打印总字符数
print("Tokens:", total_tokens)
class GPTDatasetV1(Dataset):# 继承Dataset
    def __init__(self, txt, tokenizer, max_length, stride):# 初始化函数
        self.input_ids = []
        self.target_ids = []

        # 编码文本
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})# 允许空白字符，编码为整数序列

        # 使用滑动窗口将id分块为重叠的max_length序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]# 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]# 目标序列为输入序列的后续序列
            self.input_ids.append(torch.tensor(input_chunk))# 输入序列转为tensor
            self.target_ids.append(torch.tensor(target_chunk))# 目标序列转为tensor

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
# Train/validation ratio
train_ratio = 0.90# 训练集比例
split_idx = int(train_ratio * len(text_data))# 划分训练集和验证集的索引
train_data = text_data[:split_idx]# 训练集
val_data = text_data[split_idx:]# 验证集


torch.manual_seed(123)

train_loader = create_dataloader_v1(# 创建训练集数据加载器
    train_data,
    batch_size=2,# batch大小
    max_length=GPT_CONFIG_124M["context_length"],# 最大序列长度
    stride=GPT_CONFIG_124M["context_length"],# 步长
    drop_last=True,# 丢弃最后一个不完整的batch
    shuffle=True,# 随机打乱数据
    num_workers=0# 多线程数
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],# 最大序列长度
    stride=GPT_CONFIG_124M["context_length"],# 步长
    drop_last=False,# 不丢弃最后一个不完整的batch
    shuffle=False,# 不随机打乱数据
    num_workers=0# 多线程数
)
# Sanity check

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:# 训练集token数小于上下文长度
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:# 验证集token数小于上下文长度
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")
print("Train loader:")
for x, y in train_loader:# 打印训练集数据加载器
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:# 打印验证集数据加载器
    print(x.shape, y.shape)
train_tokens = 0# 训练集token数
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()# 累计训练集token数

val_tokens = 0# 验证集token数
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()# 累计验证集token数

print("Training tokens:", train_tokens)# 打印训练集token
print("Validation tokens:", val_tokens)# 打印验证集token
print("All tokens:", train_tokens + val_tokens)# 打印所有token

def calc_loss_batch(input_batch, target_batch, model, device):# 计算给定批次的交叉熵损失
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):# 计算数据加载器中用户指定数量的批次的损失
    total_loss = 0.# 总损失
    if len(data_loader) == 0:# 数据加载器为空
        return float("nan")
    elif num_batches is None:# 未指定batches数量
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))# 限制batches数量
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:# 限制batches数量
            loss = calc_loss_batch(input_batch, target_batch, model, device)# 计算损失
            total_loss += loss.item()# 累计损失
        else:
            break
    return total_loss / num_batches# 计算平均损失
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")


model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # 禁用梯度跟踪以提高效率，因为我们还没有训练
    train_loss = calc_loss_loader(train_loader, model, device)# 计算训练集损失
    val_loss = calc_loss_loader(val_loader, model, device)# 计算验证集损失

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# 5.2训练一个LLM
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):# 训练模型
    # 初始化列表以跟踪损失和已见的token
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1# 已见token数，全局步数

    # Main training loop
    for epoch in range(num_epochs):# 训练轮数
        model.train()  # 设置模型为训练模式

        for input_batch, target_batch in train_loader:# 训练集批次
            optimizer.zero_grad()  # 重设置损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)# 计算损失
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 更新模型权重
            tokens_seen += input_batch.numel()# 累计已见token数
            global_step += 1# 累计全局步数

            # 可选评估步骤
            if global_step % eval_freq == 0:# 评估频率
                train_loss, val_loss = evaluate_model(# 评估模型
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)# 记录训练损失
                val_losses.append(val_loss)# 记录验证损失
                track_tokens_seen.append(tokens_seen)# 记录已见token数
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 打印每个epoch后样本文本
        generate_and_print_sample(# 生成并打印样本文本
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()# 设置模型为评估模式
    with torch.no_grad():# 禁用梯度跟踪以提高效率，因为还没有训练
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)# 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)# 计算验证集损失
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()# 设置模型为评估模式
    context_size = model.pos_emb.weight.shape[0]# 位置编码维度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)# 编码起始文本
    with torch.no_grad():# 禁用梯度跟踪以提高效率，因为还没训练
        token_ids = generate_text_simple(# 调用生成文本函数
            model=model, idx=encoded,# 输入起始文本
            max_new_tokens=50, context_size=context_size# 最大新token数和上下文长度
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)# 解码输出文本
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
# Note:
# Uncomment the following code to calculate the execution time
# import time
# start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)# 定义模型
model.to(device)# 移动到设备
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)# 定义优化器

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(# 训练模型
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):# 绘制损失图
    fig, ax1 = plt.subplots(figsize=(5, 3))# 创建图表

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")# 绘制训练损失
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")# 绘制验证损失
    ax1.set_xlabel("Epochs")# x轴标签
    ax1.set_ylabel("Loss")# y轴标签
    ax1.legend(loc="upper right")# 图例位置
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 只显示整数标签

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # 创建第二个x轴，共享y轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 可视化
    ax2.set_xlabel("Tokens seen")# x轴标签

    fig.tight_layout()  # 调整布局以容纳图表
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))# 转换为张量
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)# 绘制损失图
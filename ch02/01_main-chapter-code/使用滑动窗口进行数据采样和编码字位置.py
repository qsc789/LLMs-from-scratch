import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
import torch
from torch.utils.data import Dataset, DataLoader
print("PyTorch version:", torch.__version__)

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

with open("the-verdict.txt", "r", encoding="utf-8") as f:# 读取文件
    raw_text = f.read()
dataloader = create_dataloader_v1(# 创建数据加载器
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)# 创建迭代器
first_batch = next(data_iter)# 取出第一个数据
print(first_batch)

dataloader = create_dataloader_v1(# 更新数据加载器
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)# 创建迭代器
inputs, targets = next(data_iter)# 取出第一个数据
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# 编码字位置
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)# 词嵌入层
max_length = 4# 最大序列长度
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
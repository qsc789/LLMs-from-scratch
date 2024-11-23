import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import tiktoken
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt
from previous_chapters import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
# 下载并解压缩数据集
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():# 判断数据是否已经下载
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())# 保存到zip文件

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)# 解压到extracted_path

    # 添加.tsv文件后缀
    original_file_path = Path(extracted_path) / "SMSSpamCollection"# 原始文件路径
    os.rename(original_file_path, data_file_path)# 重命名为.tsv文件后缀
    print(f"File downloaded and saved as {data_file_path}")# 输出文件路径

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)# 下载并解压数据集

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])# 读取数据集
print(df["Label"].value_counts())# 输出标签分布


def create_balanced_dataset(df):
    # "spam"实例的数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机采样"ham"实例以匹配"spam"实例的数量
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 结合"ham"子集和"spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


balanced_df = create_balanced_dataset(df)# 平衡数据集
print(balanced_df["Label"].value_counts())# 输出平衡标签分布
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})# 标签转换为0/1
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)# 随机打乱数据集

    # 计算切分索引
    train_end = int(len(df) * train_frac)# 计算训练集索引
    validation_end = train_end + int(len(df) * validation_frac)# 计算验证集索引

    # Split the DataFrame
    train_df = df[:train_end]# 切分训练集
    validation_df = df[train_end:validation_end]# 切分验证集
    test_df = df[validation_end:]# 切分测试集

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)# 随机切分数据集
# 测试集大小为0.2

train_df.to_csv("train.csv", index=None)# 保存训练集
validation_df.to_csv("validation.csv", index=None)# 保存验证集
test_df.to_csv("test.csv", index=None)# 保存测试集



# 6.3 创建数据加载器
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 预先分词
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:# 如果没有设置最大长度
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果序列长度超过最大长度，截断序列
            self.encoded_texts = [
                encoded_text[:self.max_length]# 截断序列
                for encoded_text in self.encoded_texts# 遍历序列
            ]

        # 填充序列
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))# 填充序列
            for encoded_text in self.encoded_texts# 遍历序列
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]# 获取编码序列
        label = self.data.iloc[index]["Label"]# 获取标签
        return (
            torch.tensor(encoded, dtype=torch.long),# 编码序列转为tensor
            torch.tensor(label, dtype=torch.long)# 标签转为tensor
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:# 遍历序列
            encoded_length = len(encoded_text)# 获取序列长度
            if encoded_length > max_length:# 更新最大长度
                max_length = encoded_length
        return max_length
tokenizer = tiktoken.get_encoding("gpt2")# gpt2
train_dataset = SpamDataset(# 创建训练集数据加载器
    csv_file="train.csv",# 数据集路径
    max_length=None,# 最大序列长度
    tokenizer=tokenizer# 编码器
)

print(train_dataset.max_length)# 输出最大序列长度

val_dataset = SpamDataset(# 创建验证集数据加载器
    csv_file="validation.csv",# 数据集路径
    max_length=train_dataset.max_length,# 最大序列长度
    tokenizer=tokenizer# 编码器
)
test_dataset = SpamDataset(# 创建测试集数据加载器
    csv_file="test.csv",# 数据集路径
    max_length=train_dataset.max_length,# 最大序列长度
    tokenizer=tokenizer# 编码器
)


num_workers = 0# 多线程数
batch_size = 8# 批大小

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,# 数据集
    batch_size=batch_size,# 批大小
    shuffle=True,# 是否打乱数据集
    num_workers=num_workers,# 多线程数
    drop_last=True,# 是否丢弃最后一批数据
)

val_loader = DataLoader(
    dataset=val_dataset,# 数据集
    batch_size=batch_size,# 批大小
    num_workers=num_workers,# 多线程数
    drop_last=False,# 是否丢弃最后一批数据
)

test_loader = DataLoader(
    dataset=test_dataset,# 数据集
    batch_size=batch_size,# 批大小
    num_workers=num_workers,# 多线程数
    drop_last=False,# 是否丢弃最后一批数据
)
# 作为验证步骤，我们遍历数据加载器，并确保每个批次包含 8 个训练样本，其中每个训练样本由 120 个标记组成
print("Train loader:")
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
# 打印每个数据集中的批次总数
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# 6.4 使用预训练权重初始化模型
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {# 基础参数
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {# 模型参数
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])# 更新参数

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (# 验证最大序列长度是否小于模型的上下文长度
    f"Dataset length {train_dataset.max_length} exceeds model's context "# 超过模型上下文长度
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "# 重新初始化数据集
    f"`max_length={BASE_CONFIG['context_length']}`"# 设置最大序列长度为模型上下文长度
)


model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")# 获取模型大小
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")# 下载并加载模型参数

model = GPTModel(BASE_CONFIG)# 创建模型
load_weights_into_gpt(model, params)# 加载模型参数
model.eval();# 评估模式



text_1 = "Every effort moves you"

token_ids = generate_text_simple(# 生成文本
    model=model,# 模型
    idx=text_to_token_ids(text_1, tokenizer),# 文本转为token_ids
    max_new_tokens=15,# 最大新token数
    context_size=BASE_CONFIG["context_length"]# 上下文长度
)

print(token_ids_to_text(token_ids, tokenizer))# 输出生成的文本
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text_simple(# 生成文本
    model=model,# 模型
    idx=text_to_token_ids(text_2, tokenizer),# 文本转为token_ids
    max_new_tokens=23,# 最大新token数
    context_size=BASE_CONFIG["context_length"]# 上下文长度
)

print(token_ids_to_text(token_ids, tokenizer))# 输出生成的文本

# 6.5 添加分类头
print(model)# 打印模型架构
# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False
torch.manual_seed(123)

num_classes = 2# 类别数
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)# 添加分类头


for param in model.trf_blocks[-1].parameters():# 解冻最后一层参数
    param.requires_grad = True# 允许梯度更新

for param in model.final_norm.parameters():# 解冻最后一层参数
    param.requires_grad = True# 允许梯度更新
inputs = tokenizer.encode("Do you have time")# 输入文本
inputs = torch.tensor(inputs).unsqueeze(0)# 输入转为tensor
print("Inputs:", inputs)# 输出输入
print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)
with torch.no_grad():# 禁用梯度计算
    outputs = model(inputs)# 前向传播

print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)
print("Last output token:", outputs[:, -1, :])# 输出最后一个token的分类结果

# 6.6 计算分类损失和准确率
print("Last output token:", outputs[:, -1, :])# 输出最后一个token分类结果
probas = torch.softmax(outputs[:, -1, :], dim=-1)# 计算softmax概率
label = torch.argmax(probas)# 计算类别标签
print("Class label:", label.item())# 输出类别标签
logits = outputs[:, -1, :]# logits为最后一个token分类结果
label = torch.argmax(logits)# 计算类别标签
print("Class label:", label.item())# 输出类别标签
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()# 评估模式
    correct_predictions, num_examples = 0, 0# 正确预测数，样本数

    if num_batches is None:# 如果没有设置批数
        num_batches = len(data_loader)# 计算批数
    else:
        num_batches = min(num_batches, len(data_loader))# 限制最大批数
    for i, (input_batch, target_batch) in enumerate(data_loader):# 遍历数据集
        if i < num_batches:# 限制批数
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)# 转为设备

            with torch.no_grad():# 禁用梯度计算
                logits = model(input_batch)[:, -1, :]  # logits为最后一个token分类结果
            predicted_labels = torch.argmax(logits, dim=-1)# 计算类别标签

            num_examples += predicted_labels.shape[0]# 累计样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()# 累计正确预测数
        else:
            break
    return correct_predictions / num_examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 启用GPU

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# As of this writing, in PyTorch 2.4, the results obtained via CPU and MPS were identical.
# However, in earlier versions of PyTorch, you may observe different results when using MPS.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#print(f"Running on {device} device.")

model.to(device) # 转为设备

torch.manual_seed(123) # For reproducibility due to the shuffling in the training data loader

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)# 计算训练集准确率
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)# 计算验证集准确率
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)# 计算测试集准确率

print(f"Training accuracy: {train_accuracy*100:.2f}%")# 输出训练集准确率
print(f"Validation accuracy: {val_accuracy*100:.2f}%")# 输出验证集准确率
print(f"Test accuracy: {test_accuracy*100:.2f}%")# 输出测试集准确率
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)# 转为设备
    logits = model(input_batch)[:, -1, :]  # logits为最后一个token分类结果
    loss = torch.nn.functional.cross_entropy(logits, target_batch)# 计算损失
    return loss
# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.# 总损失
    if len(data_loader) == 0:# 如果数据为空
        return float("nan")
    elif num_batches is None:# 如果没有设置批数
        num_batches = len(data_loader)# 计算批数
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))# 限制最大批数1
    for i, (input_batch, target_batch) in enumerate(data_loader):# 遍历数据集
        if i < num_batches:# 限制批数
            loss = calc_loss_batch(input_batch, target_batch, model, device)# 计算损失
            total_loss += loss.item()# 累计损失
        else:
            break
    return total_loss / num_batches
with torch.no_grad(): # 禁用梯度计算
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)# 计算训练集损失
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)# 计算验证集损失
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)# 计算测试集损失

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

# 6.7 在监督数据上微调模型
# Overall the same as `train_model_simple` in chapter 5
# Same as chapter 5
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()# 评估模式
    with torch.no_grad():# 禁用梯度计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)# 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)# 计算验证集损失
    model.train()
    return train_loss, val_loss
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # 初始化列表以跟踪损失和样本数
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):# 遍历epoch
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 重置损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)# 计算损失
            loss.backward() # 计算损失梯度
            optimizer.step() # 更新模型权重
            examples_seen += input_batch.shape[0] # 累计样本数
            global_step += 1# 累计步数

            # Optional evaluation step
            if global_step % eval_freq == 0:# 评估频率
                train_loss, val_loss = evaluate_model(# 评估模型
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)# 记录损失
                val_losses.append(val_loss)# 记录损失
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)# 计算训练集准确率
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)# 计算验证集准确率
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")# 输出训练集准确率
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")# 输出验证集准确率
        train_accs.append(train_accuracy)# 记录准确率
        val_accs.append(val_accuracy)# 记录准确率

    return train_losses, val_losses, train_accs, val_accs, examples_seen
import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)# 优化器

num_epochs = 5# 训练轮数
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(# 训练分类器
    model, train_loader, val_loader, optimizer, device,# 设备
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,# 评估频率和迭代次数
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60# 计算时间（分钟）
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))# 创建图表

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")# 绘制训练损失
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")# 绘制验证损失
    ax1.set_xlabel("Epochs")# 设置横坐标标签
    ax1.set_ylabel(label.capitalize())# 设置纵坐标标签
    ax1.legend()# 显示图例

    # 创建第二个横坐标轴，用于显示样本数
    ax2 = ax1.twiny()  # 创建第二个横坐标轴，共享纵坐标周
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")# 设置横坐标标签

    fig.tight_layout()  # 调整子图间距
    plt.savefig(f"{label}-plot.pdf")# 保存图表
    plt.show()
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))# 转换为张量
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))# 转换为张量

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)# 绘制损失图表
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))# 转换为张量
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))# 转换为张量

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")# 绘制精度图表

train_accuracy = calc_accuracy_loader(train_loader, model, device)# 计算训练集准确率
val_accuracy = calc_accuracy_loader(val_loader, model, device)# 计算验证集准确率
test_accuracy = calc_accuracy_loader(test_loader, model, device)# 计算测试机准确率

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")


# 6.8 使用 LLM 作为垃圾邮件分类器
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)# 编码文本
    supported_context_length = model.pos_emb.weight.shape[0]# 获取模型支持的最大上下文长度
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]# 截断序列

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))# 填充序列
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # 添加batch维度

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()# 预测标签

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))
torch.save(model.state_dict(), "review_classifier.pth")# 保存模型
model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)# 加载模型
model.load_state_dict(model_state_dict)# 加载模型参数
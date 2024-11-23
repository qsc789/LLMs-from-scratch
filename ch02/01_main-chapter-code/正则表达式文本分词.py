
import os# 文件系统操作库
import urllib.request# URL处理库
import re # 正则表达式库


# 正则表达式文本分词
if not os.path.exists("the-verdict.txt"):# 判断文件是否存在
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)# 下载文件，并保存到指定路径

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# 使用正则表达式分割文本
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)# /s匹配空白字符
print(result)
# 对逗号和句点也进行拆分
result = re.split(r'([,.]|\s)', text)# 但会产生空白字符串，需要过滤掉
print(result)
result = [item for item in result if item.strip()]# 过滤空白字符串
print(result)

# 处理带复杂标点符号的文本
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])# 前30个token
print(len(preprocessed))# token数

# Token转化为Token ID

# 构建一个包含所有唯一 Token 的词汇表
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
vocab = {token:integer for integer,token in enumerate(all_words)}# 词汇表
for i, item in enumerate(vocab.items()):# 打印前50个词
    print(item)
    if i >= 50:
        break


class SimpleTokenizerV1:# 简单分词器
    def __init__(self, vocab):# 传入词汇表
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):# 编码
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)# 文本分词

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]# 词汇转化为ID
        return ids

    def decode(self, ids):# 解码
        text = " ".join([self.int_to_str[i] for i in ids])# ID转化为词汇
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)# 去除空格和连续标点
        return text

# 实例化分词器
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)# 编码文本
print(ids)
tokenizer.decode(ids)# 解码ID


# 添加特殊的上下文tokens
all_tokens = sorted(list(set(preprocessed)))# 所有token
all_tokens.extend(["<|endoftext|>", "<|unk|>"])# 添加<endoftext>和<unk>
vocab = {token:integer for integer,token in enumerate(all_tokens)}# 更新词汇表，包含了上下文tokens
for i, item in enumerate(list(vocab.items())[-5:]):# 打印最后5个词
    print(item)

# 我们还需要相应地调整分词器，以便它知道何时以及如何使用新的 <unk> 令牌
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed# 使用<unk>替换未知词
        ]

        ids = [self.str_to_int[s] for s in preprocessed]# 词汇转化为ID
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)# 去除多余空格和连续标点
        return text

# 实例化新分词器
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print(text)


# BytePair 编码

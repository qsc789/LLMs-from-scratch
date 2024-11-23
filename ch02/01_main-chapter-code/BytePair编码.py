import importlib
import tiktoken

# print("tiktoken version:", importlib.metadata.version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")# gpt2
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})# 允许空白字符，编码为整数序列
print(integers)
strings = tokenizer.decode(integers)# 解码为字符串
print(strings)



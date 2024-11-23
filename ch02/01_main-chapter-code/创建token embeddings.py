import torch
input_ids = torch.tensor([2, 3, 5, 1])# 输入序列
vocab_size = 6# 词表大小
output_dim = 3# 输出维度

torch.manual_seed(123)# 随机种子
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)# 词嵌入层
print(embedding_layer.weight)# 嵌入层权重
print(embedding_layer(torch.tensor([3])))# 将id为3的token转化为3维向量
print(embedding_layer(input_ids))# 嵌入4个input_ids
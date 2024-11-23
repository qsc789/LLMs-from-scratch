import torch
# 3.3.1 没有可训练权重的简单自注意力机制
# 第一步：计算非标准化的注意力分数
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # (x^2)

attn_scores_2 = torch.empty(inputs.shape[0])# 初始化attn_scores_2为inputs的形状，即6*3
for i, x_i in enumerate(inputs):# enumerate用于遍历inputs中的元素，i是索引，x_i是元素
    attn_scores_2[i] = torch.dot(x_i, query)# 计算

print(attn_scores_2)# 这个参数是没有可训练权重的简单自注意力机制的输出
res = 0.

for idx, element in enumerate(inputs[0]):# 计算第一个输入的权重
    res += inputs[0][idx] * query[idx]# idx是0,1,2，分别乘query的对应元素

print(res)# res是第一个输入的权重
print(torch.dot(inputs[0], query))# 也是第一个输入的权重，但是用了dot函数

# 第二步：注意力分数归一化
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()# 标准化，使attn_weights_2_tmp和为1
print("Attention weights:", attn_weights_2_tmp)# 输出注意力权重
print("Sum:", attn_weights_2_tmp.sum())# 和为1

# softmax函数是一个在多类分类问题中常用的激活函数，它可以将多个神经元的输出值转换为一个概率分布，类似对所有输出值归一化
def softmax_naive(x):# 定义softmax函数
    return torch.exp(x) / torch.exp(x).sum(dim=0)# dim=0表示对列求和

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)# softmax函数的输出
print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)# 直接使用softmax
print("Sum:", attn_weights_2.sum())
# 第三步：通过将嵌入的输入标记与注意力权重相乘，并将结果向量相加来计算上下文向量
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)# 置0
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i# 注意力权重与输入相乘，相加
print(context_vec_2)

# 3.3.2 计算所有输入标记的注意力权重
attn_scores = torch.empty(6, 6)# 初始化注意力分数矩阵，暴力算法
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)# 计算注意力分数
print(attn_scores)
# 也可以用矩阵乘法来计算
attn_scores = inputs @ inputs.T
print(attn_scores)
# 对矩阵归一化计算权重
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# 用步骤三计算所有上下文向量
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
# 对所有上下文向量归一化
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
print("Previous 2nd context vector:", context_vec_2)# 和矩阵第2行相等
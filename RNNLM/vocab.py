import numpy as np
import json
import random

# 加载词向量
word_embeddings = np.load('en_word_embeddings.npy')
print(word_embeddings.shape)

# # 加载词汇表
# with open('zh_vocab.json', 'r', encoding='utf-8') as f:
#     vocab = json.load(f)

# # 创建反向映射，以便通过索引查找单词
# vocab_reverse = {idx: word for word, idx in vocab.items()}

# # 随机选择10个索引
# random_indices = random.sample(range(len(word_embeddings)), 10)

# # 输出随机选中的词、索引和词向量
# print("随机选中的词及其索引和词向量:")
# for idx in random_indices:
#     word = vocab_reverse.get(idx, "未知词")
#     embedding = word_embeddings[idx]
#     print(f"词: {word}, 索引: {idx}, 词向量: {embedding}")

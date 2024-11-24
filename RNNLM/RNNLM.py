import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
import json  # 用于保存词汇表
import argparse

# 处理文本数据,准备数据集
class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_length=5):
        self.vocab = vocab
        self.seq_length = seq_length
        self.data = self.prepare_data(text)

    def prepare_data(self, text):
        # 按行分割文本
        lines = text.splitlines()
        indices = []
        
        # 逐行处理每个句子
        for line in lines:
            # 将每行中的单词转换为索引
            line_indices = [self.vocab[word] for word in line.split() if word in self.vocab]
            # 将索引按 seq_length 切分
            indices.extend([line_indices[i:i + self.seq_length] for i in range(len(line_indices) - self.seq_length)])
        
        return indices

    def __len__(self):
        return len(self.data) - 1  # 确保 len(data) - 1 是有效的

    def __getitem__(self, idx):
        if idx >= len(self.data) - 1:  # 添加边界检查
            raise IndexError("Index out of range")
        
        input_seq = self.data[idx]
        target = self.data[idx + 1][-1]  # 预测下一个单词
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# 定义 RNNLM 模型
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # 嵌入层的权重矩阵：形状为 (vocab_size, embedding_dim)，每一行对应词汇表中的一个单词的嵌入向量。
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.fc(rnn_out[:, -1, :])  # 只使用最后一个时间步的输出
        return out

# 从文件中读取文本数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='RNNLM')
parser.add_argument("--data_type", type=str, default="zh", help="Data type, zh or en.")
args = parser.parse_args()
data_type = args.data_type

# 数据准备
file_path = f'data/{args.data_type}.txt'
text = load_data(file_path)

# 构建词汇表
words = text.split()
vocab = Counter(words)
vocab = {word: idx for idx, (word, _) in enumerate(vocab.items())}
vocab_size = len(vocab)

# 保存词汇表

with open(f'{args.data_type}_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

# 超参数
embedding_dim = 50
hidden_dim = 100
seq_length = 5
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# 数据集和数据加载器
dataset = TextDataset(text, vocab, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
model = RNNLM(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 使用 tqdm 包装 dataloader，显示进度条
    for inputs, targets in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存词向量
word_embeddings = model.embedding.weight.data.numpy()
np.save(f'{args.data_type}_word_embeddings.npy', word_embeddings)

# 形状为 (vocab_size, embedding_dim)
# print(word_embeddings.shape)

print("训练完成，词向量已保存。")

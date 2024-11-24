# RNNLM模型词向量

https://github.com/Cbdlll/NLP-Basic-Class/tree/master/exp2

## 模型结构

```python
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
```

实现了一个基于 LSTM 的循环神经网络语言模型：

- 嵌入层，用于将词汇表中的单词映射到一个连续的向量空间
- LSTM 层，处理嵌入后的序列数据并捕捉时间依赖性
- 全连接层，将 LSTM 的输出转换为词汇表大小的预测概率分布

 `forward` 方法中，输入经过嵌入层后，传递给 LSTM，并只使用最后一个时间步的输出进行词汇预测

## 训练方法

```python
# 训练模型
for epoch in range(num_epochs):
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
```

使用**BPTT（Backpropagation Through Time）**训练方法，在 **BP** 的基础上发展而来的，主要用于处理具有时间依赖性的序列数据。

由于 **RNN** 的结构涉及时间序列数据，**BPTT** 通过将时间展开为多个时间步来计算梯度。在每个时间步上，网络的状态依赖于前一个时间步的状态，因此需要对整个序列进行反向传播。

## 思考

词嵌入（Embedding），将文本数据映射到高维空间，使得相似的单词在嵌入空间中距离较近，这样可以捕捉到词语之间的语义关系，可以通过计算余弦相似度来判断语义相关性。

RNNLM词向量方法中，有一个权重矩阵，形状是（vocab_size, embedding_dim），每一行代表词汇表中一个单词的嵌入表示。

输入文本传入模型后，经过Embedding层转换成为词向量表示，更符合模型的输入要求，适合于后续的模型处理。

在训练过程中，计算损失并进行反向传播时，因为Embedding 层的权重矩阵在处理输入文本时会影响输出所以也会被计算梯度，从而被训练优化更新。这意味着词向量会随着模型学习逐步优化，从而更好地表示词语的语义信息。
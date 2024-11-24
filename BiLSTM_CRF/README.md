# Bi-LSTM+CRF for NER

**<font color=red>Code and model.pth：https://github.com/Cbdlll/NLP_Basic/tree/master/BiLSTM_CRF</font>**

## Model Structure

```python
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, pretrained_embeddings=None):
        super(BiLSTM_CRF, self).__init__()
        
        # 嵌入层
        if pretrained_embeddings is not None:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Bi-LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        # 用于从 Bi-LSTM 输出转换到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # CRF 层
        self.crf = CRF(tagset_size, batch_first=True)
        
    def forward(self, sentences):
        # 获取嵌入表示
        embeddings = self.embeddings(sentences)
        # 通过 Bi-LSTM
        lstm_out, _ = self.lstm(embeddings)
        # 通过线性层转换为标签空间
        lstm_out = self.hidden2tag(lstm_out)
        return lstm_out
    
    def loss(self, sentences, tags):
        # 通过 CRF 计算损失
        lstm_out = self.forward(sentences)
        loss = -self.crf(lstm_out, tags)
        return loss
    
    def predict(self, sentences):
        # 通过 CRF 预测标签
        lstm_out = self.forward(sentences)
        prediction = self.crf.decode(lstm_out)
        return prediction
```

## Hyperparameters

```python
train_corpus_path = "data/train_corpus.txt"
train_label_path = "data/train_label.txt"
test_corpus_path = "data/test_corpus.txt"
test_label_path = "data/test_label.txt"
batch_size = 32
embedding_dim = 100
hidden_dim = 128
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Process Data

创建词汇表和标签表

```python
train_data = load_data(train_corpus_path, train_label_path)
test_data = load_data(test_corpus_path, test_label_path)

# 构建词表和标签表
word2idx, tag2idx = build_vocab(train_data)
# 将数据编码为索引
train_sentences, train_labels = encode_data(train_data, word2idx, tag2idx)
test_sentences, test_labels = encode_data(test_data, word2idx, tag2idx)
# 创建 DataLoader
train_loader = create_dataloader(train_sentences, train_labels, batch_size)
test_loader = create_dataloader(test_sentences, test_labels, batch_size)
```

## Trainer

```python
class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
    
    def train(self):
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            train_loop = tqdm(self.train_loader, total= len(self.train_loader), desc=f"Training Epoch {epoch+1}/{self.epochs}")
            for sentences, labels in train_loop:
                sentences, labels = sentences.to(self.device), labels.to(self.device)

                # 前向传播
                loss = self.model.loss(sentences, labels)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                running_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())
            self.scheduler.step(running_loss)
            
        torch.save(self.model.state_dict(), 'BiLSTM_CRF.pth')
        return running_loss
```

## Train

```python
    model = BiLSTM_CRF(len(word2idx), len(tag2idx), embedding_dim, hidden_dim).to(device)
    model_path = 'BiLSTM_CRF.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
        
        trainer = Trainer(model, train_loader, test_loader, optimizer, scheduler, num_epochs, device)
        trainer.train()
```

## Training Results

```shell
wjg@14x:~/project/BiLSTM_CRF$ python main.py 
Loading data...
Building vocabularies...
Encoding data...
Creating DataLoaders...
Initializing model...
Training Epoch 1/10: 100%|███████████████████████| 1584/1584 [00:56<00:00, 28.20it/s, loss=4.7]
Training Epoch 2/10: 100%|██████████████████████| 1584/1584 [00:52<00:00, 30.44it/s, loss=4.15]
Training Epoch 3/10: 100%|██████████████████████| 1584/1584 [00:51<00:00, 30.63it/s, loss=2.94]
Training Epoch 4/10: 100%|██████████████████████| 1584/1584 [00:54<00:00, 28.95it/s, loss=2.05]
Training Epoch 5/10: 100%|█████████████████████| 1584/1584 [00:59<00:00, 26.74it/s, loss=0.354]
Training Epoch 6/10: 100%|█████████████████████| 1584/1584 [00:59<00:00, 26.56it/s, loss=0.159]
Training Epoch 7/10: 100%|█████████████████████| 1584/1584 [00:59<00:00, 26.53it/s, loss=0.136]
Training Epoch 8/10: 100%|████████████████████| 1584/1584 [00:59<00:00, 26.53it/s, loss=0.0225]
Training Epoch 9/10: 100%|███████████████████| 1584/1584 [00:59<00:00, 26.74it/s, loss=0.00757]
Training Epoch 10/10: 100%|████████████████████| 1584/1584 [00:59<00:00, 26.55it/s, loss=0.131]
```

## Evaluator

使用`remove_o()`去除`O`标签，实现方法：去除`true_labels`中的`O`，并将`pred_labels`对应索引处的预测标签去除

使用`zero_division=0`，应对去除`O`标签后，在计算中会出现分母为0的情况。

```python
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model, test_loader, device, tag2idx):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.tag2idx = tag2idx

    def remove_o(self, pred_labels, true_labels):
        """
        去除 'O' 类别标签，并返回去除 'O' 后的预测和真实标签。
        :param pred_labels: 预测标签列表
        :param true_labels: 真实标签列表
        :return: 去除 'O' 后的预测和真实标签
        """
        # 'O' 的索引
        o_idx = self.tag2idx.get('O', None)  # 确保 'O' 在标签中
        if o_idx is None:
            return pred_labels, true_labels

        # 过滤掉 'O' 标签的部分
        pred_labels_filtered = []
        true_labels_filtered = []
        
        # 遍历预测和真实标签，对齐处理
        for p, t in zip(pred_labels, true_labels):
            if t != o_idx:  # 如果真实标签不是 'O'
                pred_labels_filtered.append(p)  # 保留
                true_labels_filtered.append(t)  # 保留
        return pred_labels_filtered, true_labels_filtered



    def evaluate_with_O(self):
        self.model.eval()
        
        all_pred_labels = []  # 存储所有预测的标签
        all_true_labels = []  # 存储所有真实的标签

        with torch.no_grad():
            for sentences, labels in self.test_loader:
                sentences, labels = sentences.to(self.device), labels.to(self.device)

                # 获取预测结果
                predicted_labels = self.model.predict(sentences)

                for pred_seq, true_seq in zip(predicted_labels, labels):
                    # 去掉填充部分并比较
                    pred_seq = pred_seq[:len(true_seq)]
                    true_seq = true_seq[:len(pred_seq)]

                    all_pred_labels.extend(pred_seq)
                    all_true_labels.extend(true_seq)

        # 计算准确率、召回率、F1值
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')

        # 打印各项指标
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Precision: {precision * 100:.2f}%')
        print(f'Recall: {recall * 100:.2f}%')
        print(f'F1 Score: {f1 * 100:.2f}%')

        return accuracy, precision, recall, f1


    def evaluate_without_O(self):
        self.model.eval()
        
        all_pred_labels = []  # 存储所有预测的标签
        all_true_labels = []  # 存储所有真实的标签

        with torch.no_grad():
            for sentences, labels in self.test_loader:
                sentences, labels = sentences.to(self.device), labels.to(self.device)

                # 获取预测结果
                predicted_labels = self.model.predict(sentences)

                for pred_seq, true_seq in zip(predicted_labels, labels):
                    # 去掉填充部分并比较
                    pred_seq = pred_seq[:len(true_seq)]
                    true_seq = true_seq[:len(pred_seq)]

                    # 去除 'O' 类别标签
                    pred_seq_filtered, true_seq_filtered = self.remove_o(pred_seq, true_seq)

                    all_pred_labels.extend(pred_seq_filtered)
                    all_true_labels.extend(true_seq_filtered)

        # 计算准确率、召回率、F1值
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')

        # 打印各项指标
        print(f'Accuracy (without O): {accuracy * 100:.2f}%')
        print(f'Precision (without O): {precision * 100:.2f}%')
        print(f'Recall (without O): {recall * 100:.2f}%')
        print(f'F1 Score (without O): {f1 * 100:.2f}%')

        return accuracy, precision, recall, f1
```

## Evaluate

```python
evaluator = Evaluator(model, test_loader, device, tag2idx)
evaluator.evaluate_with_O()
evaluator.evaluate_without_O()
```

## Evaluate Results

模型的整体表现优秀，高准确率和高F1分数表示模型在预测实体时，误差较少，对实体的捕捉比较成功。

```shell
Accuracy: 98.34%
Precision: 98.30%
Recall: 98.34%
F1 Score: 98.31%

Accuracy (without O): 98.40%
Precision (without O): 99.28%
Recall (without O): 98.40%
F1 Score (without O): 98.80%
```

## Predict One

```python
def predict_one(model, prim_sentence, word2idx, idx2tag):
    # 在每个字中间加入空格
    sentence = ' '.join(prim_sentence)
    # 将句子转换为词索引
    sentence_indices = [word2idx.get(word, word2idx['<UNK>']) for word in sentence.split()]
    print(f'sentence_indices: {sentence_indices}')
    
    sentence_tensor = torch.tensor(sentence_indices).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentence_tensor = sentence_tensor.to(device)
    
    # 预测标签
    predicted_labels = model.predict(sentence_tensor)
    print(f'predicted_labels_idx: {predicted_labels[0]}')
    # 将预测的标签索引转换为标签名称
    predicted_labels_names = [list(idx2tag.items())[idx][1] for idx in predicted_labels[0]]
    
    result = list(zip(prim_sentence, predicted_labels_names))
    print(f'Predicted results: {result}')
```

```python
sentence = '我在北京，参观天安门'
predict_one(model, sentence, word2idx, idx2tag)
```

```python
sentence_indices: [983, 665, 1022, 1621, 974, 1242, 1446, 66, 1820, 3899]
predicted_labels_idx: [2, 2, 1, 6, 2, 2, 2, 3, 0, 0]
Predicted results: [('我', 'O'), ('在', 'O'), ('北', 'B-LOC'), ('京', 'I-LOC'), ('，', 'O'), ('参', 'O'), ('观', 'O'), ('天', 'B-ORG'), ('安', 'I-ORG'), ('门', 'I-ORG')]
```


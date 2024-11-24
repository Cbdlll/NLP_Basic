import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

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

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from BiLSTM_CRF import BiLSTM_CRF
from Process_Data import create_dataloader, build_vocab, encode_data, save_vocab, load_vocab
from Load_Data import load_data
from Trainer import Trainer
from Evaluator import Evaluator
from Predict_one import predict_one
import warnings
warnings.filterwarnings("ignore", message=".*where received a uint8 condition tensor.*")

def main():
    # 配置
    train_corpus_path = "data/train_corpus.txt"
    train_label_path = "data/train_label.txt"
    test_corpus_path = "data/test_corpus.txt"
    test_label_path = "data/test_label.txt"
    batch_size = 32
    embedding_dim = 100
    hidden_dim = 128
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    print("Loading data...")
    train_data = load_data(train_corpus_path, train_label_path)
    test_data = load_data(test_corpus_path, test_label_path)

    # print(f"Sample train data: {train_data[2]}")
    
    # 构建词表和标签表
    print("Building vocabularies...")
    vocab_path = 'vocab/word2idx.json'
    tag_path = 'vocab/tag2idx.json'
    idx2tag_path = 'vocab/idx2tag.json'
    
    if os.path.exists(vocab_path) and os.path.exists(tag_path) and os.path.exists(idx2tag_path):
        word2idx, tag2idx, idx2tag = load_vocab(vocab_path, tag_path, idx2tag_path)
    else:
        word2idx, tag2idx = build_vocab(train_data)
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}

        save_vocab(word2idx, tag2idx, idx2tag, vocab_path, tag_path, idx2tag_path)
    
    # print(f"word2idx: {list(word2idx.items())}")
    # print(f"tag2idx: {list(tag2idx.items())}")
    # print(f"idx2tag: {list(idx2tag.items())}")
     
    # 将数据编码为索引
    print("Encoding data...")
    train_sentences, train_labels = encode_data(train_data, word2idx, tag2idx)
    test_sentences, test_labels = encode_data(test_data, word2idx, tag2idx)
    # 训练集中的第一条句子和标签
    # print(f"Sample sentence (encoded): {train_sentences[0]}")
    # print(f"Sample labels (encoded): {train_labels[0]}")

    # 创建 DataLoader
    print("Creating DataLoaders...")
    train_loader = create_dataloader(train_sentences, train_labels, batch_size)
    test_loader = create_dataloader(test_sentences, test_labels, batch_size)

    print("Initializing model...")
    
    model = BiLSTM_CRF(len(word2idx), len(tag2idx), embedding_dim, hidden_dim).to(device)
    model_path = 'BiLSTM_CRF.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
        
        trainer = Trainer(model, train_loader, test_loader, optimizer, scheduler, num_epochs, device)
        trainer.train()
    
    # evaluator = Evaluator(model, test_loader, device, tag2idx)
    # evaluator.evaluate_with_O()
    # evaluator.evaluate_without_O()
    
    sentence = '我在北京，参观天安门'
    predict_one(model, sentence, word2idx, idx2tag)
    
if __name__ == "__main__":
    main()
    

from torch.utils.data import DataLoader, TensorDataset
import torch
import json

def create_dataloader(sentences, labels, batch_size=32):
    """
    创建 PyTorch DataLoader。
    :param sentences: 编码后的句子列表
    :param labels: 编码后的标签列表
    :param batch_size: 批大小
    :return: DataLoader 对象
    """
    # 填充序列
    max_len = max(len(s) for s in sentences)
    sentences_padded = [s + [0] * (max_len - len(s)) for s in sentences]  # 填充 0
    labels_padded = [l + [0] * (max_len - len(l)) for l in labels]        # 填充 0
    
    # 转为 Tensor
    sentences_tensor = torch.tensor(sentences_padded, dtype=torch.long)
    labels_tensor = torch.tensor(labels_padded, dtype=torch.long)
    
    # 创建 DataLoader
    dataset = TensorDataset(sentences_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def build_vocab(data):
    """
    根据训练数据构建词表和标签表。
    :param data: 包含 (sentence, labels) 的列表
    :return: word2idx, tag2idx
    """
    word_set = set()
    tag_set = set()
    
    for sentence, labels in data:
        word_set.update(sentence)
        tag_set.update(labels)
    
    # 添加特殊 tokens
    word2idx = {word: idx for idx, word in enumerate(word_set, start=2)}
    word2idx["<PAD>"] = 0  # Padding token
    word2idx["<UNK>"] = 1  # Unknown token

    tag2idx = {tag: idx for idx, tag in enumerate(tag_set)}
    return word2idx, tag2idx

def encode_data(data, word2idx, tag2idx):
    """
    将句子和标签转换为索引表示。
    :param data: 包含 (sentence, labels) 的列表
    :param word2idx: 词汇表
    :param tag2idx: 标签表
    :return: (encoded_sentences, encoded_labels)
    """
    encoded_sentences = []
    encoded_labels = []

    for sentence, labels in data:
        encoded_sentences.append([word2idx.get(word, word2idx["<UNK>"]) for word in sentence])
        encoded_labels.append([tag2idx[label] for label in labels])
    
    return encoded_sentences, encoded_labels

    # 定义保存函数
def save_vocab(word2idx, tag2idx, idx2tag, vocab_path, tag_path, idx2tag_path):
    # 保存 word2idx 到文件
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=4)
    
    # 保存 tag2idx 到文件
    with open(tag_path, 'w', encoding='utf-8') as f:
        json.dump(tag2idx, f, ensure_ascii=False, indent=4)

    # 保存 idx2tag 到文件
    with open(idx2tag_path, 'w', encoding='utf-8') as f:
        json.dump(idx2tag, f, ensure_ascii=False, indent=4)

# 定义加载函数
def load_vocab(vocab_path, tag_path, idx2tag_path):
    # 加载 word2idx
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)

    # 加载 tag2idx
    with open(tag_path, 'r', encoding='utf-8') as f:
        tag2idx = json.load(f)

    # 加载 idx2tag
    with open(idx2tag_path, 'r', encoding='utf-8') as f:
        idx2tag = json.load(f)

    return word2idx, tag2idx, idx2tag
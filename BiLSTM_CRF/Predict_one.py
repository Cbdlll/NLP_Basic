import torch
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
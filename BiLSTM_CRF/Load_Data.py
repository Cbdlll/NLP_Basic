def load_data(corpus_path, label_path):
    """
    加载语料和标签文件，并确保它们一一对应。
    :param corpus_path: 语料文件路径
    :param label_path: 标签文件路径
    :return: 一个包含 (sentence, labels) 的列表
    """
    sentences = []
    labels = []
    
    # 读取 corpus 文件
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉换行符并拆分为单词列表
            sentences.append(line.strip().split())

    # 读取 label 文件
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉换行符并拆分为标签列表
            labels.append(line.strip().split())

    # 确保语料和标签一一对应
    assert len(sentences) == len(labels), "语料和标签行数不一致！"
    return list(zip(sentences, labels))

def load_combined_data(file_path):
    """
    加载包含文本和标签的整合文件，每行表示一个 token 和其标签。
    :param file_path: 文件路径
    :return: 一个包含 (sentence, labels) 的列表
    """
    data = []
    sentence = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉换行符
            line = line.strip()
            if not line:
                # 如果遇到空行，表示当前句子结束
                if sentence and labels:
                    data.append((sentence, labels))
                    sentence = []
                    labels = []
            else:
                # 将行分为 token 和标签
                token, label = line.split()
                sentence.append(token)
                labels.append(label)

    # 添加最后一个句子（如果文件结尾没有空行）
    if sentence and labels:
        data.append((sentence, labels))

    return data


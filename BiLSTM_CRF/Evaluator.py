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
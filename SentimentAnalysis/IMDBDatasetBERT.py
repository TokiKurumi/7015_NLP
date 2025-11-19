# datasets/bert_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class IMDBDatasetBERT(Dataset):
    """
    BERT专用的IMDB情感分析数据集
    """

    def __init__(self, texts, labels, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        # 转换为tensor
        self.labels_tensor = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        返回BERT格式的数据样本
        """
        text = str(self.texts[idx])
        label = self.labels_tensor[idx]

        if not text or text.strip() == "":
            text = "[EMPTY_TEXT]"

        # BERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

    def get_stats(self):
        """返回数据集统计信息"""
        return {
            'total_samples': len(self),
            'positive_samples': sum(self.labels),
            'negative_samples': len(self.labels) - sum(self.labels),
            'positive_ratio': sum(self.labels) / len(self.labels),
            'max_length': self.max_length,
        }

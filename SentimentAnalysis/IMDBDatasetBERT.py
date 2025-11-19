# datasets/bert_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import CONFIG
import os
class IMDBDatasetBERT(Dataset):
    """
    BERT专用的IMDB情感分析数据集
    """

    def __init__(self, texts, labels, max_length=512):
        self.texts = texts
        self.labels = labels
        # 尝试从本地加载，如果失败则从网络下载
        local_path = CONFIG['BERT_LOCAL_PATH']

        try:
            # 检查本地是否有词汇表文件
            if os.path.exists(os.path.join(local_path, "vocab.txt")):
                print(f"从本地加载BERT tokenizer: {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            else:
                print("本地BERT词汇表不存在，从网络下载...")
                self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['BERT_MODEL_NAME'])
        except Exception as e:
            print(f"加载BERT tokenizer失败: {e}")
            print("从网络下载BERT tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['BERT_MODEL_NAME'])
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

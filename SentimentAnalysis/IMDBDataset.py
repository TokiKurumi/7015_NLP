import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class IMDBDataset(Dataset):
    """
    自定义IMDB情感分析数据集
    """

    def __init__(self, sequences, labels, vocab=None):
        """
        参数:
        sequences: 文本序列列表 [[1, 23, 45, ...], [2, 67, 89, ...]]
        labels: 标签列表 [1, 0, 1, ...]
        vocab: 词汇表 (可选，用于调试)
        """
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab

        # 转换为tensor
        self.sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        self.labels_tensor = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        返回一个数据样本
        """
        sequence = self.sequences_tensor[idx]
        label = self.labels_tensor[idx]

        return {
            'sequence': sequence,
            'label': label,
            'length': torch.tensor(len([x for x in sequence if x != 0]), dtype=torch.long)  # 实际长度(非padding)
        }

    def get_stats(self):
        """返回数据集统计信息"""
        return {
            'total_samples': len(self),
            'positive_samples': sum(self.labels),
            'negative_samples': len(self.labels) - sum(self.labels),
            'sequence_length': self.sequences_tensor.shape[1],
            'positive_ratio': sum(self.labels) / len(self.labels)
        }
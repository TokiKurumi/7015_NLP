# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout


class SentimentLSTM(nn.Module):
    """
    LSTM情感分析模型
    支持两种嵌入方式：随机初始化 和 预训练词向量
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, dropout, pad_idx, embedding_matrix=None,
                 freeze_embedding=False):
        super().__init__()

        # 嵌入层
        if embedding_matrix is not None:
            # 使用预训练词向量
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix,
                padding_idx=pad_idx,
                freeze=freeze_embedding  # 是否冻结词向量
            )
        else:
            # 随机初始化嵌入
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=pad_idx
            )

        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,  # 可以先尝试单向
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # 分类层
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # 存储参数
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, text, text_lengths):
        # text shape: [batch_size, seq_len]

        # 嵌入层
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, seq_len, embedding_dim]

        # 打包序列（处理变长序列）
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM层
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        last_hidden = hidden[-1]

        # 解包序列
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(
        #    packed_output, batch_first=True
        #)
        #batch_size = output.size(0)
        #seq_len = output.size(1)

        # 创建索引，确保不超出序列范围
        #indices = text_lengths - 1
        #indices = torch.clamp(indices, 0, seq_len - 1)  # 限制在有效范围内

        # 使用高级索引获取最后一个有效时间步
        #last_output = output[torch.arange(batch_size), indices]
        # 使用最后一个时间步的隐藏状态
        # output shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [num_layers, batch_size, hidden_dim]

        # 取最后一个有效时间步的输出
        #last_output = output[range(len(output)), text_lengths - 1, :]

        # 分类
        return self.fc(self.dropout(last_hidden))
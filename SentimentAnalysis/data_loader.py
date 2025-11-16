from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from SentimentAnalysis.IMDBDataset import IMDBDataset


# 1. 构建词汇表 (Vocabulary Building)
def build_vocab(texts, min_freq=2):
    word_counts = Counter()
    # 第一步：统计所有单词的出现频率
    for text in texts:
        words = text.split()  # 假设clean_text已经是分词后的形式
        word_counts.update(words)

    # # 第二步：创建词汇映射字典：word -> index
    vocab = {'<PAD>': 0, '<UNK>': 1} # 特殊token
    idx = 2 # 从2开始，因为0和1已经被特殊token占用
    # 第三步：过滤低频词并分配索引
    for word, count in word_counts.items():
        if count >= min_freq: # 只保留出现次数≥min_freq的词
            vocab[word] = idx
            idx += 1
    return vocab


# 2. 文本序列化 (Text to Sequence)
def text_to_sequence(text, vocab, max_len=None):
    words = text.split()
    # 第二步：词汇表查找，将单词转换为索引, word=找到对应索引,未找到则为'<UNK>'
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]

    if max_len:
        # 截断：保留前max_len个词
        if len(sequence) > max_len:
            sequence = sequence[:max_len]  # 截断
        else:
            # 填充：在末尾添加<PAD>直到达到max_len
            sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))  # 填充
    return sequence


def to_tensor(data):
    return torch.tensor(data, dtype=torch.long)

# 3. 准备预训练词向量 (如GloVe)
def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    """加载GloVe词向量并创建嵌入矩阵"""
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # 创建嵌入矩阵
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            # 对于不在GloVe中的词，随机初始化
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return torch.tensor(embedding_matrix, dtype=torch.float)


def create_data_loaders(train_df, val_df, test_df, batch_size=32, shuffle=True):
    """
    创建支持动态padding的DataLoader

    参数:
    train_df, val_df, test_df: 包含'sequence'和'label'列的DataFrame
    batch_size: 批量大小
    shuffle: 是否打乱训练数据
    """

    # 确保输入是DataFrame且包含必要的列
    for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name}必须是pandas DataFrame")
        if 'sequence' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"{name}必须包含'sequence'和'label'列")

    print("正在创建支持动态padding的DataLoader...")

    # 创建Dataset实例
    train_dataset = IMDBDataset(
        sequences=train_df['sequence'].tolist(),
        labels=train_df['label'].tolist()
    )

    val_dataset = IMDBDataset(
        sequences=val_df['sequence'].tolist(),
        labels=val_df['label'].tolist()
    )

    test_dataset = IMDBDataset(
        sequences=test_df['sequence'].tolist(),
        labels=test_df['label'].tolist()
    )

    # 使用collate_fn支持动态padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # 使用传入的shuffle参数
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        collate_fn=collate_fn,
        num_workers=0
    )

    # 打印统计信息
    print("=== 动态padding DataLoader 统计信息 ===")
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    print(f"批次大小: {batch_size}")
    print(f"训练集是否打乱: {shuffle}")

    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """
    自定义批次处理函数，支持动态padding和按长度排序
    """
    sequences = [item['sequence'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]

    # 按序列长度降序排序（有助于LSTM的pack_padded_sequence）
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sequences = [sequences[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]

    # 转换为tensor
    sequences_tensor = torch.stack(sequences)
    labels_tensor = torch.stack(labels)
    lengths_tensor = torch.stack(lengths)

    return {
        'sequences': sequences_tensor,
        'labels': labels_tensor,
        'lengths': lengths_tensor
    }


def main():
    filepath = os.path.join('data', 'vocab.txt')

    # 读取处理后的数据
    train_df = pd.read_csv('imdb_train_processed.csv')
    test_df = pd.read_csv('imdb_test_processed.csv')

    if not os.path.exists(filepath):
        vocab = build_vocab(train_df['clean_text'])
        print(f"词汇表大小: {len(vocab)}")
        save_vocabulary(vocab, filepath)
        print(f"保存完成: {filepath}")
    else:
        vocab = load_vocabulary(filepath)

    # 转换为序列
    max_sequence_length = 256  # 根据你的数据调整
    train_df['sequence'] = train_df['clean_text'].apply(
        lambda x: text_to_sequence(x, vocab, max_sequence_length)
    )
    test_df['sequence'] = test_df['clean_text'].apply(
        lambda x: text_to_sequence(x, vocab, max_sequence_length)
    )


    # 直接使用DataFrame进行划分，而不是转换为列表
    train_processed, val_processed = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label']
    )

    # 重置索引
    train_processed = train_processed.reset_index(drop=True)
    val_processed = val_processed.reset_index(drop=True)
    test_processed = test_df.reset_index(drop=True)

    print(f"训练集: {len(train_processed)}, 验证集: {len(val_processed)}, 测试集: {len(test_processed)}")

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_data_loaders(
        train_processed, val_processed, test_processed,
        batch_size=32,
    )

    # 测试DataLoader
    print("\n=== DataLoader 测试 ===")
    for i, batch in enumerate(train_loader):
        if i == 0:  # 只查看第一个批次
            sequences = batch['sequences']
            labels = batch['labels']
            lengths = batch['lengths']

            print(f"批次序列形状: {sequences.shape}")
            print(f"批次标签形状: {labels.shape}")
            print(f"批次长度形状: {lengths.shape}")
            print(f"序列示例: {sequences[0][:10]}...")
            print(f"标签示例: {labels[:5]}")
            print(f"长度示例: {lengths[:5]}")
            break

    return train_loader, val_loader, test_loader , len(vocab)


def save_vocabulary(vocab, filepath):
    """
    保存词汇表到JSON文件
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"词汇表已保存到: {filepath}")

def load_vocabulary(filepath):
    """
    从JSON文件加载词汇表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"词汇表已加载，大小: {len(vocab)}")
    return vocab

if __name__ == "__main__":
    main()

# 使用示例（如果你有GloVe文件）
# glove_path = 'glove.6B.100d.txt'
# embedding_matrix = load_glove_embeddings(glove_path, vocab)
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
import json
import re
from torch.utils.data import DataLoader
from SentimentAnalysis.IMDBDataset import IMDBDataset
from SentimentAnalysis.IMDBDatasetBERT import IMDBDatasetBERT
from config import CONFIG


def clean_text(text):
    """
    text pre processing function
    :param text:
    text(str)
    :return:
    str
    """
    if not isinstance(text, str):
        return ""

    #1. remove HTML labels
    text = re.sub(r'<[^>]+>', '', text)
    #2. translate to lower
    text = text.lower()
    #3. remove URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    #4. remove e-mail
    text = re.sub(r'\S+@\S+', '', text)
    #5. keep letter, number, and remove special words
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    #6. remove the space not need
    text = re.sub(r'\s+', ' ', text).strip()

    return text


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

    print(f"Vocabulary has been built: {len(vocab)} words, remove {len(word_counts) - len(vocab) + 2} words ")
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
    print(f"Loading GloVe embeddings from {glove_path}")
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                if len(values) < embedding_dim + 1:  # 确保格式正确
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"错误: GloVe文件不存在: {glove_path}")
        return None
    except Exception as e:
        print(f"加载GloVe文件时出错: {e}")
        return None

    print(f"Successfully loaded GloVe embeddings {len(embeddings_index)}")
    # 创建嵌入矩阵
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    found_words = 0
    missing_words = []

    for word, idx in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Find pre-train vectors
            embedding_matrix[idx] = embedding_vector
            found_words += 1
        elif word.lower() in embeddings_index:
            # try lower letter
            embedding_matrix[idx] = embeddings_index[word.lower()]
            found_words += 1
        else:
            # 对于不在GloVe中的词，随机初始化
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
            if word not in ['<PAD>', '<UNK>']:
                missing_words.append(word)
    # Analysis report
    coverage = found_words / len(vocab) * 100
    print(f"词汇表覆盖率: {found_words}/{len(vocab)} ({coverage:.2f}%)")

    if missing_words and len(missing_words) > 10:
        print(f"前10个缺失词示例: {missing_words[:10]}")

    return torch.tensor(embedding_matrix, dtype=torch.float)


def process_dataframe(df, text_column='review', label_column='sentiment', clean_text_func=None):
    """
    处理DataFrame，进行文本清理和标签转换
    参数:
        df (pd.DataFrame): 输入数据框
        text_column (str): 文本列名
        label_column (str): 标签列名
        clean_text_func (function): 文本清理函数
    返回:
        pd.DataFrame: 处理后的数据框
    """
    df_processed = df.copy()

    # 文本清理
    if clean_text_func and text_column in df_processed.columns:
        print("正在进行文本清理...")
        df_processed['clean_text'] = df_processed[text_column].apply(clean_text_func)
        print(f"文本清理完成，示例: {df_processed['clean_text'].iloc[0][:100]}...")

    # 标签转换（如果标签是字符串）
    if label_column in df_processed.columns:
        if df_processed[label_column].dtype == 'object':
            # 假设标签是 'positive'/'negative' 或类似格式
            label_mapping = {'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0}
            df_processed['label'] = df_processed[label_column].map(label_mapping)
            # 处理未映射的标签
            df_processed['label'] = df_processed['label'].fillna(-1).astype(int)
        else:
            df_processed['label'] = df_processed[label_column]

    # 移除无效数据
    original_len = len(df_processed)
    df_processed = df_processed[df_processed['label'].isin([0, 1])]
    if 'clean_text' in df_processed.columns:
        df_processed = df_processed[df_processed['clean_text'].str.len() > 0]

    print(f"数据清洗完成: 移除 {original_len - len(df_processed)} 个无效样本")

    return df_processed


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


def create_bert_data_loaders(train_df, val_df, test_df, batch_size=16,
                             num_workers=None,max_length = 512 ):
    """
    创建BERT专用的DataLoader
    """
    print("正在创建BERT DataLoader...")

    if num_workers is None:
        num_workers = 4 if torch.cuda.is_available() else 2

    # 确保输入是DataFrame且包含必要的列
    for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name}必须是pandas DataFrame")
        if 'clean_text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"{name}必须包含'clean_text'和'label'列")

    print("正在创建BERT DataLoader...")

    # 创建BERT Dataset实例
    train_dataset = IMDBDatasetBERT(
        texts=train_df['clean_text'].tolist(),
        labels=train_df['label'].tolist(),
        max_length=max_length
    )

    val_dataset = IMDBDatasetBERT(
        texts=val_df['clean_text'].tolist(),
        labels=val_df['label'].tolist(),
        max_length=max_length
    )

    test_dataset = IMDBDatasetBERT(
        texts=test_df['clean_text'].tolist(),
        labels=test_df['label'].tolist(),
        max_length=max_length
    )

    # 使用优化配置创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True,  # 加速GPU数据传输
        persistent_workers=num_workers > 0,  # 减少进程创建开销
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    print("=== BERT DataLoader 统计信息 ===")
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    print(f"批次大小: {batch_size}")

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


def analyze_dataset(df, name="数据集"):
    """
    分析数据集统计信息
    """
    print(f"\n=== {name} 分析 ===")
    print(f"总样本数: {len(df)}")
    print(f"正样本数: {sum(df['label'] == 1)}")
    print(f"负样本数: {sum(df['label'] == 0)}")

    if 'clean_text' in df.columns:
        text_lengths = df['clean_text'].str.split().str.len()
        print(f"平均文本长度: {text_lengths.mean():.2f}")
        print(f"最大文本长度: {text_lengths.max()}")
        print(f"最小文本长度: {text_lengths.min()}")

def load_data():
    train_file = CONFIG['TRAIN_FILE']
    test_file = CONFIG['TEST_FILE']
    vocab_file = CONFIG['VOCAB_FILE']

    # 检查数据文件是否存在
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"错误: 数据文件不存在")
        print(f"请确保以下文件存在:")
        print(f"- {train_file}")
        print(f"- {test_file}")
        return None, None, None, 0

    print("正在加载数据...")

    # 读取原始数据
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, 0

    # 处理数据（文本清理和标签转换）
    train_df = process_dataframe(train_df, clean_text_func=clean_text)
    test_df = process_dataframe(test_df, clean_text_func=clean_text)

    # 分析数据集
    analyze_dataset(train_df, "训练集")
    analyze_dataset(test_df, "测试集")


    if not os.path.exists(vocab_file):
        vocab = build_vocab(train_df['clean_text'])
        print(f"词汇表大小: {len(vocab)}")
        save_vocabulary(vocab, vocab_file)
        print(f"保存完成: {vocab_file}")
    else:
        vocab = load_vocabulary(vocab_file)

    # 转换为序列
    max_sequence_length = 256  # 根据你的数据调整
    print(f"\n正在将文本转换为序列 (最大长度: {max_sequence_length})...")
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
    print(f"\n数据划分完成:")
    print(f"训练集: {len(train_processed)}, 验证集: {len(val_processed)}, 测试集: {len(test_processed)}")

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_data_loaders(
        train_processed, val_processed, test_processed,
        batch_size=32,
        shuffle=True
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

    return train_loader, val_loader, test_loader, len(vocab), vocab


def load_bert_data(batch_size=None, max_length=None):
    """
    加载BERT专用的数据
    """
    train_file = CONFIG['TRAIN_FILE']
    test_file = CONFIG['TEST_FILE']

    # 使用配置中的默认值或传入的参数
    if batch_size is None:
        batch_size = CONFIG['BERT_BATCH_SIZE']
    if max_length is None:
        max_length = CONFIG['BERT_MAX_LENGTH']

    # 检查数据文件是否存在
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"错误: 数据文件不存在")
        print(f"请确保以下文件存在:")
        print(f"- {train_file}")
        print(f"- {test_file}")
        return None, None, None

    print("正在加载BERT数据...")

    # 读取原始数据
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None

    # 处理数据（文本清理和标签转换）
    train_df = process_dataframe(train_df, clean_text_func=clean_text)
    test_df = process_dataframe(test_df, clean_text_func=clean_text)

    # 分析数据集
    analyze_dataset(train_df, "BERT训练集")
    analyze_dataset(test_df, "BERT测试集")

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

    print(f"\nBERT数据划分完成:")
    print(f"训练集: {len(train_processed)}, 验证集: {len(val_processed)}, 测试集: {len(test_processed)}")

    # 创建BERT DataLoader
    train_loader, val_loader, test_loader = create_bert_data_loaders(
        train_processed, val_processed, test_processed,
        batch_size=batch_size,
        max_length=max_length
    )

    # 测试BERT DataLoader
    print("\n=== BERT DataLoader 测试 ===")
    for i, batch in enumerate(train_loader):
        if i == 0:  # 只查看第一个批次
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            print(f"输入ID形状: {input_ids.shape}")
            print(f"注意力掩码形状: {attention_mask.shape}")
            print(f"标签形状: {labels.shape}")
            print(f"标签示例: {labels[:5]}")
            break

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab_size = load_data()
    if train_loader is not None:
        print(f"\n数据加载成功! 词汇表大小: {vocab_size}")
    else:
        print("数据加载失败!")

# 使用示例（如果你有GloVe文件）
# glove_path = 'glove.6B.100d.txt'
# embedding_matrix = load_glove_embeddings(glove_path, vocab)
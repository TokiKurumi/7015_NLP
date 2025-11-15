"""
嵌入层工具函数
负责词向量的加载、初始化和分析
支持两种嵌入方式：随机初始化和预训练词向量（GloVe）
"""

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import os
import json


def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    """
    从GloVe文件加载预训练词向量并创建嵌入矩阵

    参数:
        glove_path (str): GloVe词向量文件路径
        vocab (dict): 词汇表字典 {word: index}
        embedding_dim (int): 词向量维度，必须与GloVe文件维度匹配

    返回:
        torch.Tensor: 嵌入矩阵 [vocab_size, embedding_dim]
    """
    print(f"正在从 {glove_path} 加载GloVe {embedding_dim}D 词向量...")

    # 检查文件是否存在
    if not os.path.exists(glove_path):
        raise FileNotFoundError(f"GloVe文件不存在: {glove_path}")

    # 读取GloVe文件，构建词向量索引
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            if len(values) < embedding_dim + 1:  # 确保行格式正确
                continue
            word = values[0]
            coefs = np.asarray(values[1:embedding_dim + 1], dtype='float32')
            embeddings_index[word] = coefs

    print(f"成功加载 {len(embeddings_index)} 个预训练词向量")

    # 构建嵌入矩阵
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found_words = 0
    missing_words = []

    for word, idx in vocab.items():
        if word in embeddings_index:
            # 找到预训练向量
            embedding_matrix[idx] = embeddings_index[word]
            found_words += 1
        elif word.lower() in embeddings_index:
            # 尝试小写形式
            embedding_matrix[idx] = embeddings_index[word.lower()]
            found_words += 1
        else:
            # 未找到的词，使用随机初始化
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
            if word not in ['<PAD>', '<UNK>']:  # 特殊token不计入缺失
                missing_words.append(word)

    # 分析报告
    coverage = found_words / vocab_size * 100
    print(f"词汇表覆盖率: {found_words}/{vocab_size} ({coverage:.2f}%)")

    # 打印前10个缺失的高频词（如果有的话）
    if missing_words and len(missing_words) > 10:
        print(f"前10个缺失词示例: {missing_words[:10]}")

    return torch.tensor(embedding_matrix, dtype=torch.float)


def create_random_embedding_matrix(vocab_size, embedding_dim=100, padding_idx=0):
    """
    创建随机初始化的嵌入矩阵

    参数:
        vocab_size (int): 词汇表大小
        embedding_dim (int): 词向量维度
        padding_idx (int): 填充token的索引

    返回:
        nn.Embedding: 随机初始化的嵌入层
    """
    print(f"创建随机初始化嵌入层: [{vocab_size}, {embedding_dim}]")

    # 创建嵌入层
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx
    )

    # 使用Xavier初始化（可选，但通常效果更好）
    nn.init.xavier_uniform_(embedding.weight)

    # 将padding_idx对应的权重设为0
    if padding_idx is not None:
        with torch.no_grad():
            embedding.weight[padding_idx] = torch.zeros(embedding_dim)

    return embedding


def get_embedding_layer(vocab_size, embedding_dim=100, padding_idx=0,
                        glove_path=None, vocab=None, freeze_embedding=False):
    """
    统一的嵌入层创建函数，支持随机初始化和预训练两种方式

    参数:
        vocab_size (int): 词汇表大小
        embedding_dim (int): 词向量维度
        padding_idx (int): 填充token的索引
        glove_path (str): GloVe文件路径，如果为None则使用随机初始化
        vocab (dict): 词汇表字典，仅在预训练模式下需要
        freeze_embedding (bool): 是否冻结嵌入层权重

    返回:
        nn.Embedding: 嵌入层
    """
    if glove_path is not None and vocab is not None:
        # 使用预训练词向量
        print("使用预训练GloVe词向量...")
        embedding_matrix = load_glove_embeddings(glove_path, vocab, embedding_dim)

        # 创建嵌入层
        embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            padding_idx=padding_idx,
            freeze=freeze_embedding
        )

        print(f"嵌入层已{'冻结' if freeze_embedding else '微调'}")
    else:
        # 使用随机初始化
        print("使用随机初始化嵌入层...")
        embedding = create_random_embedding_matrix(
            vocab_size, embedding_dim, padding_idx
        )

    return embedding


def analyze_vocabulary(texts, vocab=None, top_k=20):
    """
    分析词汇表统计信息

    参数:
        texts (list): 文本列表
        vocab (dict): 词汇表字典，如果为None则从文本构建
        top_k (int): 显示最高频词的数量

    返回:
        dict: 词汇统计信息
    """
    # 分词并统计词频
    word_freq = Counter()
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            word_freq.update(words)

    # 如果提供了词汇表，分析覆盖率
    vocab_analysis = {}
    if vocab is not None:
        vocab_words = set(vocab.keys())
        text_words = set(word_freq.keys())

        # 计算覆盖率
        covered_words = vocab_words.intersection(text_words)
        coverage = len(covered_words) / len(vocab_words) * 100

        vocab_analysis = {
            'vocab_size': len(vocab_words),
            'text_vocab_size': len(text_words),
            'coverage': coverage,
            'oov_words': len(text_words - vocab_words)
        }

    # 构建统计信息
    stats = {
        'total_unique_words': len(word_freq),
        'total_tokens': sum(word_freq.values()),
        'avg_word_frequency': np.mean(list(word_freq.values())),
        'top_words': word_freq.most_common(top_k),
        'vocab_analysis': vocab_analysis
    }

    return stats


def save_embedding_info(embedding_layer, vocab, filepath):
    """
    保存嵌入层信息用于分析和调试

    参数:
        embedding_layer (nn.Embedding): 嵌入层
        vocab (dict): 词汇表
        filepath (str): 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 获取嵌入权重
    embedding_weights = embedding_layer.weight.data.cpu().numpy()

    # 构建词向量字典
    word_vectors = {}
    for word, idx in vocab.items():
        if idx < len(embedding_weights):
            word_vectors[word] = embedding_weights[idx].tolist()

    # 保存为JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(word_vectors, f, ensure_ascii=False, indent=2)

    print(f"嵌入层信息已保存到: {filepath}")


def compare_embeddings(embedding1, embedding2, vocab, top_k=10):
    """
    比较两个嵌入层的相似度

    参数:
        embedding1, embedding2: 两个嵌入层
        vocab (dict): 词汇表
        top_k (int): 显示最相似和最不相似的词数量

    返回:
        dict: 比较结果
    """
    # 获取权重矩阵
    weights1 = embedding1.weight.data.cpu().numpy()
    weights2 = embedding2.weight.data.cpu().numpy()

    # 计算余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = []
    for word, idx in vocab.items():
        if idx < min(len(weights1), len(weights2)):
            vec1 = weights1[idx].reshape(1, -1)
            vec2 = weights2[idx].reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]
            similarities.append((word, similarity))

    # 排序
    similarities.sort(key=lambda x: x[1])

    # 返回结果
    return {
        'most_similar': similarities[-top_k:][::-1],  # 最相似的
        'least_similar': similarities[:top_k],  # 最不相似的
        'avg_similarity': np.mean([s[1] for s in similarities])
    }


def download_glove_if_needed(glove_dir='data/glove', dimension=100):
    """
    如果需要，自动下载GloVe词向量（可选功能）

    参数:
        glove_dir (str): 保存GloVe文件的目录
        dimension (int): 词向量维度 (50, 100, 200, 300)

    返回:
        str: GloVe文件路径
    """
    import urllib.request
    import zipfile

    os.makedirs(glove_dir, exist_ok=True)

    glove_filename = f'glove.6B.{dimension}d.txt'
    glove_path = os.path.join(glove_dir, glove_filename)
    zip_path = os.path.join(glove_dir, 'glove.6B.zip')

    # 检查文件是否已存在
    if os.path.exists(glove_path):
        print(f"GloVe文件已存在: {glove_path}")
        return glove_path

    # 下载GloVe
    url = f'https://nlp.stanford.edu/data/glove.6B.zip'
    print(f"下载GloVe词向量...")

    try:
        urllib.request.urlretrieve(url, zip_path)

        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)

        # 删除zip文件
        os.remove(zip_path)

        print(f"GloVe词向量下载完成: {glove_path}")
        return glove_path

    except Exception as e:
        print(f"下载GloVe失败: {e}")
        print("请手动下载: https://nlp.stanford.edu/projects/glove/")
        return None


# 测试函数
def test_embedding_utils():
    """测试嵌入工具函数"""
    print("=== 测试嵌入工具函数 ===")

    # 创建测试词汇表
    test_vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2,
        'movie': 3,
        'is': 4,
        'great': 5,
        'bad': 6,
        'good': 7,
        'excellent': 8,
        'terrible': 9
    }

    # 测试随机嵌入
    print("\n1. 测试随机嵌入:")
    random_embedding = create_random_embedding_matrix(
        vocab_size=len(test_vocab),
        embedding_dim=50
    )
    print(f"随机嵌入层形状: {random_embedding.weight.shape}")

    # 测试词汇分析
    print("\n2. 测试词汇分析:")
    test_texts = [
        "the movie is great",
        "the movie is bad",
        "this is a good movie",
        "excellent film",
        "terrible acting"
    ]
    stats = analyze_vocabulary(test_texts, test_vocab)
    print(f"总唯一词数: {stats['total_unique_words']}")
    print(f"高频词: {stats['top_words'][:5]}")

    # 测试统一嵌入层创建
    print("\n3. 测试统一嵌入层创建:")
    embedding = get_embedding_layer(
        vocab_size=len(test_vocab),
        embedding_dim=50,
        padding_idx=0,
        glove_path=None,  # 使用随机初始化
        vocab=test_vocab
    )
    print(f"嵌入层创建成功: {embedding.weight.shape}")

    print("\n所有测试完成!")


if __name__ == "__main__":
    test_embedding_utils()
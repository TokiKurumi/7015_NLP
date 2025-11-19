"""
基于notebook的数据分析和可视化模块
将Jupyter notebook中的分析代码转换为可重用的Python模块
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from collections import Counter
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
sns.set(font_scale=1.2)


class IMDBDataAnalyzer:
    """
    IMDB数据集分析器
    基于原始notebook代码，提供数据分析和可视化功能
    """

    def __init__(self, data_path="aclImdb"):
        self.data_path = data_path
        self.train_df = None
        self.test_df = None
        self.vocab = None
        self.stats = {}

    def load_imdb_from_local(self):
        """
        从本地加载IMDB数据集
        增强版本，包含更多错误检查和诊断信息
        """

        def load_split(split_path):
            texts = []
            labels = []

            # 检查路径是否存在
            if not os.path.exists(split_path):
                print(f"警告: 路径不存在 {split_path}")
                return pd.DataFrame({'text': texts, 'label': labels})

            # Load positive reviews
            pos_path = os.path.join(split_path, 'pos')
            if os.path.exists(pos_path):
                pos_files = glob.glob(os.path.join(pos_path, '*.txt'))
                print(f"找到 {len(pos_files)} 个正面评论文件")

                # 限制文件数量用于测试
                max_files = min(1000, len(pos_files))
                for i, file_path in enumerate(pos_files[:max_files]):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                            labels.append(1)  # Positive label is 1
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

                    # 显示进度
                    if (i + 1) % 500 == 0:
                        print(f"已加载 {i + 1} 个正面评论文件")
            else:
                print(f"警告: 正面评论路径不存在 {pos_path}")

            # Load negative reviews
            neg_path = os.path.join(split_path, 'neg')
            if os.path.exists(neg_path):
                neg_files = glob.glob(os.path.join(neg_path, '*.txt'))
                print(f"找到 {len(neg_files)} 个负面评论文件")

                # 限制文件数量用于测试
                max_files = min(1000, len(neg_files))
                for i, file_path in enumerate(neg_files[:max_files]):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                            labels.append(0)  # Negative label is 0
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

                    # 显示进度
                    if (i + 1) % 500 == 0:
                        print(f"已加载 {i + 1} 个负面评论文件")
            else:
                print(f"警告: 负面评论路径不存在 {neg_path}")

            print(f"成功加载 {len(texts)} 条评论")
            return pd.DataFrame({'text': texts, 'label': labels})

        train_path = os.path.join(self.data_path, 'train')
        test_path = os.path.join(self.data_path, 'test')

        print(f"数据路径: {self.data_path}")
        print(f"训练集路径: {train_path}")
        print(f"测试集路径: {test_path}")

        # 检查路径是否存在
        if not os.path.exists(self.data_path):
            print(f"错误: 数据根路径不存在: {self.data_path}")
            print("请检查数据路径是否正确，或下载IMDB数据集")
            return None, None

        print("Loading training set...")
        self.train_df = load_split(train_path)
        print("Loading test set...")
        self.test_df = load_split(test_path)

        if self.train_df is not None and len(self.train_df) > 0:
            print(f"Training set shape: {self.train_df.shape}")
            print(f"Training set label distribution:\n{self.train_df['label'].value_counts().sort_index()}")
        else:
            print("警告: 训练集为空")

        if self.test_df is not None and len(self.test_df) > 0:
            print(f"Test set shape: {self.test_df.shape}")
            print(f"Test set label distribution:\n{self.test_df['label'].value_counts().sort_index()}")
        else:
            print("警告: 测试集为空")

        return self.train_df, self.test_df

    def clean_text(self, text):
        """
        改进的文本清洗函数 - 更彻底的清理
        """
        if not isinstance(text, str):
            return ""

        # 1. 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 2. 移除URL
        text = re.sub(r'http\S+', '', text)
        # 3. 移除@提及
        text = re.sub(r'@\w+', '', text)

        # 4. 小写化
        text = text.lower()

        # 5. 处理省略号（多个连续的点）
        text = re.sub(r'\.{2,}', ' ', text)

        # 6. 移除所有标点符号，但保留单词内部的连字符和撇号
        # 这个正则表达式会保留字母、数字、空格和单词内部的连字符(-)和撇号(')
        text = re.sub(r'[^\w\s\'-]', ' ', text)

        # 7. 处理单词开头或结尾的标点
        # 移除单词开头的标点（连字符和撇号除外）
        text = re.sub(r'\s+[-\\']', ' ', text)
        # 移除单词结尾的标点（连字符和撇号除外）
        text = re.sub(r'[-\\']\s+', ' ', text)

        # 8. 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def apply_text_cleaning(self):
        """
        应用文本清洗
        """
        print("Starting text cleaning...")

        # 检查数据是否存在
        if self.train_df is None or len(self.train_df) == 0:
            print("错误: 训练集为空，无法进行文本清洗")
            return None, None

        if self.test_df is None or len(self.test_df) == 0:
            print("错误: 测试集为空，无法进行文本清洗")
            return None, None

        # Apply cleaning to training and test sets
        self.train_df['clean_text'] = self.train_df['text'].apply(self.clean_text)
        self.test_df['clean_text'] = self.test_df['text'].apply(self.clean_text)

        print("Text cleaning completed")

        # 安全地显示清洗前后的对比
        print("\nComparison before and after cleaning:")
        print("=" * 80)

        # 只显示前3个样本，确保索引存在
        num_samples = min(3, len(self.train_df))
        for i in range(num_samples):
            try:
                print(f"\nSample {i + 1} (Label: {'Positive' if self.train_df['label'].iloc[i] == 1 else 'Negative'}):")
                print(f"Original text: {self.train_df['text'].iloc[i][:100]}...")
                print(f"Cleaned text: {self.train_df['clean_text'].iloc[i][:100]}...")
                print(
                    f"Length change: {len(self.train_df['text'].iloc[i])} → {len(self.train_df['clean_text'].iloc[i])} characters")
                print("-" * 80)
            except Exception as e:
                print(f"显示样本 {i + 1} 时出错: {e}")
                break

        return self.train_df, self.test_df

    def tokenize_word_level(self, text):
        """单词级别分词 - 简单空格分词"""
        import string
        if not isinstance(text, str):
            return []

        # 使用空格分词，并过滤掉空字符串
        tokens = text.split()

        # 进一步清理：移除只包含标点的token
        cleaned_tokens = []
        for token in tokens:
            # 移除token开头和结尾的标点
            clean_token = token.strip(string.punctuation)
            if clean_token:  # 确保不是空字符串
                cleaned_tokens.append(clean_token)

        return cleaned_tokens

    def tokenize_character_level(self, text):
        """字符级别分词"""
        if not isinstance(text, str):
            return []

        # 移除空格，将文本拆分为字符
        text_no_spaces = text.replace(' ', '')
        characters = list(text_no_spaces)

        return characters

    def tokenize_subword_level(self, text, min_n=2, max_n=4):
        """子词级别分词 - 使用n-gram方法"""
        if not isinstance(text, str):
            return []

        # 首先进行单词级别分词
        words = self.tokenize_word_level(text)
        subwords = []

        for word in words:
            # 对于短单词，直接保留
            if len(word) <= min_n:
                subwords.append(word)
                continue

            # 对于长单词，使用n-gram拆分
            # 生成不同长度的n-gram
            for n in range(min_n, min(max_n + 1, len(word) + 1)):
                for i in range(len(word) - n + 1):
                    subword = word[i:i + n]
                    subwords.append(subword)

            # 同时保留完整的单词
            subwords.append(word)

        return subwords

    def tokenize_text(self, text, tokenization_type='word', **kwargs):
        """
        根据选择的分词类型进行分词
        """
        if tokenization_type == 'word':
            return self.tokenize_word_level(text)
        elif tokenization_type == 'char':
            return self.tokenize_character_level(text)
        elif tokenization_type == 'subword':
            min_n = kwargs.get('min_n', 2)
            max_n = kwargs.get('max_n', 4)
            return self.tokenize_subword_level(text, min_n=min_n, max_n=max_n)
        else:
            raise ValueError(f"不支持的分词类型: {tokenization_type}。请选择 'word', 'char' 或 'subword'")

    def apply_tokenization(self, tokenization_type='word', **kwargs):
        """
        应用分词 - 支持多种分词类型
        tokenization_type: 分词类型，可选 'word', 'char', 'subword'
        **kwargs: 分词参数，如subword分词的min_n和max_n
        """
        print(f"Applying {tokenization_type} tokenization...")

        # 检查数据是否存在
        if self.train_df is None or len(self.train_df) == 0:
            print("错误: 训练集为空，无法进行分词")
            return None, None

        if self.test_df is None or len(self.test_df) == 0:
            print("错误: 测试集为空，无法进行分词")
            return None, None

        # Apply tokenization based on type
        if tokenization_type == 'subword':
            self.train_df['tokens'] = self.train_df['clean_text'].apply(
                lambda x: self.tokenize_text(x, tokenization_type, **kwargs)
            )
            self.test_df['tokens'] = self.test_df['clean_text'].apply(
                lambda x: self.tokenize_text(x, tokenization_type, **kwargs)
            )
        else:
            self.train_df['tokens'] = self.train_df['clean_text'].apply(
                lambda x: self.tokenize_text(x, tokenization_type)
            )
            self.test_df['tokens'] = self.test_df['clean_text'].apply(
                lambda x: self.tokenize_text(x, tokenization_type)
            )

        print("Tokenization completed!")

        # 安全地显示示例
        try:
            if len(self.train_df) > 0:
                print(f"Sample tokens: {self.train_df['tokens'].iloc[0][:10]}...")

                # Show example
                sample_text = self.train_df['clean_text'].iloc[0]
                print(f"\nExample:")
                print(f"Original text: {sample_text[:50]}...")
                print(f"Tokens: {self.train_df['tokens'].iloc[0][:10]}...")
        except Exception as e:
            print(f"显示分词示例时出错: {e}")

        return self.train_df, self.test_df

    def calculate_basic_statistics(self):
        """
        基本统计分析
        """
        print("Basic statistical analysis")
        print("=" * 50)

        # 检查数据是否存在
        if self.train_df is None or len(self.train_df) == 0:
            print("错误: 训练集为空，无法进行统计分析")
            return {}

        if self.test_df is None or len(self.test_df) == 0:
            print("错误: 测试集为空，无法进行统计分析")
            return {}

        # Calculate review length
        self.train_df['length'] = self.train_df['clean_text'].apply(len)
        self.test_df['length'] = self.test_df['clean_text'].apply(len)

        # Statistics
        print("Training set statistics:")
        train_avg_len = self.train_df['length'].mean()
        train_median_len = self.train_df['length'].median()
        train_max_len = self.train_df['length'].max()
        train_min_len = self.train_df['length'].min()

        print(f"  Average length: {train_avg_len:.2f} characters")
        print(f"  Median length: {train_median_len:.2f} characters")
        print(f"  Longest review: {train_max_len} characters")
        print(f"  Shortest review: {train_min_len} characters")

        print("\nTest set statistics:")
        test_avg_len = self.test_df['length'].mean()
        test_median_len = self.test_df['length'].median()
        test_max_len = self.test_df['length'].max()
        test_min_len = self.test_df['length'].min()

        print(f"  Average length: {test_avg_len:.2f} characters")
        print(f"  Median length: {test_median_len:.2f} characters")
        print(f"  Longest review: {test_max_len} characters")
        print(f"  Shortest review: {test_min_len} characters")

        # Statistics by sentiment
        print("\nStatistics by sentiment (training set):")
        sentiment_stats = {}
        for label in [0, 1]:
            label_data = self.train_df[self.train_df['label'] == label]['length']
            if len(label_data) > 0:
                avg_length = label_data.mean()
                sentiment = 'Negative' if label == 0 else 'Positive'
                print(f"  Label {label} ({sentiment}): Average {avg_length:.2f} characters")
                sentiment_stats[f'label_{label}_avg_length'] = avg_length

        # Store statistics
        self.stats.update({
            'train_avg_length': train_avg_len,
            'train_median_length': train_median_len,
            'train_max_length': train_max_len,
            'train_min_length': train_min_len,
            'test_avg_length': test_avg_len,
            'test_median_length': test_median_len,
            'test_max_length': test_max_len,
            'test_min_length': test_min_len,
            **sentiment_stats
        })

        return self.stats

    def build_vocabulary(self, min_freq=2):
        """
        构建词汇表
        """
        print("Building vocabulary...")

        # 检查数据是否存在
        if self.train_df is None or len(self.train_df) == 0:
            print("错误: 训练集为空，无法构建词汇表")
            return {}

        # Count frequency of all words
        word_counts = Counter()
        all_texts = self.train_df['clean_text'].tolist()

        for text in all_texts:
            words = text.split()  # Split by space
            word_counts.update(words)

        print(f"Found {len(word_counts)} unique words")
        print(f"Top 10 most frequent words: {word_counts.most_common(10)}")

        # Build vocabulary: only keep words with sufficient frequency
        self.vocab = {}
        index = 2  # Start from 2, 0 and 1 are for special tokens

        # Add special tokens
        self.vocab['<PAD>'] = 0  # Padding token
        self.vocab['<UNK>'] = 1  # Unknown word token

        # Add regular words
        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab[word] = index
                index += 1

        print(
            f"Vocabulary size: {len(self.vocab)} (including {len(self.vocab) - 2} regular words and 2 special tokens)")

        # Store vocabulary stats
        self.stats['vocab_size'] = len(self.vocab)
        self.stats['top_words'] = word_counts.most_common(20)

        print("\nVocabulary examples (first 20):")
        for i, (word, idx) in enumerate(list(self.vocab.items())[:20]):
            print(f"  {word}: {idx}")

        return self.vocab

    def convert_to_sequences(self, max_sequence_length=None):
        """
        将文本转换为数值序列
        """
        print("Converting text to numerical sequences...")

        # 检查词汇表是否存在
        if self.vocab is None:
            print("错误: 词汇表为空，请先构建词汇表")
            return None, None

        def text_to_sequence(text, vocab, max_length=None):
            """
            Convert text to numerical sequence
            """
            words = text.split()
            sequence = []

            for word in words:
                # If word is in vocabulary, use its index; otherwise use <UNK> token
                sequence.append(vocab.get(word, vocab['<UNK>']))

            # Truncate if max_length is specified
            if max_length and len(sequence) > max_length:
                sequence = sequence[:max_length]

            return sequence

        # 安全地测试转换
        try:
            if len(self.train_df) > 0:
                sample_text = self.train_df['clean_text'].iloc[0]
                sample_sequence = text_to_sequence(sample_text, self.vocab)
                print(f"Sample text: {sample_text[:100]}...")
                print(f"Converted sequence (first 20): {sample_sequence[:20]}")
                print(f"Sequence length: {len(sample_sequence)}")
        except Exception as e:
            print(f"测试序列转换时出错: {e}")

        # Apply to all data
        self.train_df['sequence'] = self.train_df['clean_text'].apply(
            lambda x: text_to_sequence(x, self.vocab, max_sequence_length)
        )
        self.test_df['sequence'] = self.test_df['clean_text'].apply(
            lambda x: text_to_sequence(x, self.vocab, max_sequence_length)
        )

        # Calculate sequence length statistics
        sequence_lengths = [len(seq) for seq in self.train_df['sequence']]
        if max_sequence_length is None:
            max_sequence_length = int(np.percentile(sequence_lengths, 95))

        self.stats['avg_sequence_length'] = np.mean(sequence_lengths)
        self.stats['max_sequence_length'] = max_sequence_length
        self.stats['sequence_lengths'] = sequence_lengths

        print("Text to sequence conversion completed!")
        print(f"Average sequence length: {self.stats['avg_sequence_length']:.1f}")
        print(f"Selected max sequence length: {max_sequence_length}")

        return self.train_df, self.test_df

    def create_visualizations(self, save_path=None):
        """
        创建数据可视化
        """
        print("Data visualization")
        print("=" * 50)

        # 检查数据是否存在
        if self.train_df is None or len(self.train_df) == 0:
            print("错误: 训练集为空，无法创建可视化")
            return None

        if self.test_df is None or len(self.test_df) == 0:
            print("错误: 测试集为空，无法创建可视化")
            return None

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Training set length distribution
        axes[0, 0].hist(self.train_df['length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.train_df['length'].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {self.train_df["length"].mean():.1f}')
        axes[0, 0].set_xlabel('Review length (character count)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Training set review length distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Test set length distribution
        axes[0, 1].hist(self.test_df['length'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(self.test_df['length'].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {self.test_df["length"].mean():.1f}')
        axes[0, 1].set_xlabel('Review length (character count)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Test set review length distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Sentiment label distribution
        label_counts_train = self.train_df['label'].value_counts().sort_index()
        label_counts_test = self.test_df['label'].value_counts().sort_index()

        x = np.arange(2)
        width = 0.35

        axes[1, 0].bar(x - width / 2, label_counts_train.values, width, label='Training set', alpha=0.7)
        axes[1, 0].bar(x + width / 2, label_counts_test.values, width, label='Test set', alpha=0.7)
        axes[1, 0].set_xlabel('Sentiment label')
        axes[1, 0].set_ylabel('Sample count')
        axes[1, 0].set_title('Sentiment label distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['Negative (0)', 'Positive (1)'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(label_counts_train.values):
            axes[1, 0].text(i - width / 2, v + 100, str(v), ha='center')
        for i, v in enumerate(label_counts_test.values):
            axes[1, 0].text(i + width / 2, v + 100, str(v), ha='center')

        # 4. Boxplot of review length by sentiment
        box_data = [self.train_df[self.train_df['label'] == 0]['length'],
                    self.train_df[self.train_df['label'] == 1]['length']]
        axes[1, 1].boxplot(box_data, labels=['Negative (0)', 'Positive (1)'])
        axes[1, 1].set_ylabel('Review length (character count)')
        axes[1, 1].set_title('Review length distribution by sentiment')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()
        print("Visualization completed")

        return fig

    def save_processed_data(self, train_path='imdb_train_processed.csv',
                            test_path='imdb_test_processed.csv'):
        """
        保存处理后的数据
        """
        # 检查数据是否存在
        if self.train_df is None or len(self.train_df) == 0:
            print("错误: 训练集为空，无法保存")
            return None, None

        if self.test_df is None or len(self.test_df) == 0:
            print("错误: 测试集为空，无法保存")
            return None, None

        save_processed = input("Save processed data? (y/n): ").lower().strip()
        if save_processed == 'y':
            self.train_df.to_csv(train_path, index=False)
            self.test_df.to_csv(test_path, index=False)
            print(f"Processed data saved as '{train_path}' and '{test_path}'")
            return train_path, test_path
        else:
            print("Data not saved")
            return None, None

    def run_complete_analysis(self, max_sequence_length=256, save_plots=True,
                             tokenization_type='word', tokenization_params=None):
        """
        运行完整的数据分析流程
        """
        print("=== Running Complete IMDB Data Analysis ===")

        if tokenization_params is None:
            tokenization_params = {}

        try:
            # 1. 加载数据
            print("步骤 1/7: 加载数据...")
            train_df, test_df = self.load_imdb_from_local()

            if train_df is None or len(train_df) == 0:
                print("错误: 数据加载失败，请检查数据路径")
                return {}

            # 2. 文本清洗
            print("步骤 2/7: 文本清洗...")
            self.apply_text_cleaning()

            # 3. 分词
            print("步骤 3/7: 分词...")
            self.apply_tokenization(tokenization_type, **tokenization_params)

            # 4. 统计分析
            print("步骤 4/7: 统计分析...")
            self.calculate_basic_statistics()

            # 5. 构建词汇表
            print("步骤 5/7: 构建词汇表...")
            self.build_vocabulary()

            # 6. 转换为序列
            print("步骤 6/7: 转换为序列...")
            self.convert_to_sequences(max_sequence_length)

            # 7. 可视化
            print("步骤 7/7: 创建可视化...")
            if save_plots:
                self.create_visualizations('data_distribution.png')

            # 8. 保存数据（可选）
            train_path, test_path = self.save_processed_data()

            print("\n数据预处理检查完成!")
            print("\n建议的下一步:")
            print("1. 检查数据质量和清洗效果")
            print("2. 确认数据分布符合预期")
            print("3. 如果满意，可以开始模型训练")

            return self.stats

        except Exception as e:
            print(f"数据分析过程中出错: {e}")
            import traceback
            print("详细错误信息:")
            traceback.print_exc()
            return {}


def run_my_analysis(data_path="aclImdb", tokenization_type='word', tokenization_params=None):
    """
    运行完整分析流程的便捷函数
    在main.py中调用
    """
    print("=== Running My Notebook Analysis ===")

    try:
        analyzer = IMDBDataAnalyzer(data_path)
        stats = analyzer.run_complete_analysis(
            tokenization_type=tokenization_type,
            tokenization_params=tokenization_params
        )

        if stats:
            print("\n=== Analysis Summary ===")
            print(f"训练样本: {len(analyzer.train_df) if analyzer.train_df is not None else 0}")
            print(f"测试样本: {len(analyzer.test_df) if analyzer.test_df is not None else 0}")
            print(f"词汇表大小: {stats.get('vocab_size', 'N/A')}")
            print(f"平均评论长度: {stats.get('train_avg_length', 0):.1f} 字符")
            print(f"最大序列长度: {stats.get('max_sequence_length', 0)}")
        else:
            print("分析失败，未生成统计数据")

        return stats

    except Exception as e:
        print(f"运行分析时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}


def find_imdb_data():
    """
    尝试自动找到IMDB数据集的路径
    """
    possible_paths = [
        "aclImdb",
        "./aclImdb",
        "../aclImdb",
        "../../aclImdb",
        "./data/aclImdb",
        "../data/aclImdb",
        "C:/Users/DR/Desktop/HKBU文件/AAA上课文件/7015/※Group Project/NLP/7015_NLP-main/aclImdb"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到IMDB数据集: {path}")
            return path

    print("未找到IMDB数据集，请手动下载并放置在项目目录下")
    return None


if __name__ == "__main__":
    # 直接运行时的测试代码
    print("测试数据分析模块...")

    # 尝试自动找到数据路径
    data_path = find_imdb_data()
    if data_path is None:
        data_path = "aclImdb"  # 默认路径

    # 测试不同的分词类型
    stats = run_my_analysis(data_path, tokenization_type='word')

    # 也可以测试其他分词类型：
    # stats = run_my_analysis(data_path, tokenization_type='char')
    # stats = run_my_analysis(data_path, tokenization_type='subword', tokenization_params={'min_n': 2, 'max_n': 4})

    if stats:
        print("\n分析成功完成!")
    else:
        print("\n分析失败，请检查错误信息")
# config.py
import os

# 全局配置
CONFIG = {
    # 模型参数
    'EMBEDDING_DIM': 300,
    'HIDDEN_DIM': 256,
    'OUTPUT_DIM': 1,
    'N_LAYERS': 2,
    'DROPOUT': 0.5,
    'PAD_IDX': 0,

    # 训练参数
    'EPOCHS': 1,
    'BERT_EPOCHS': 1,
    'BATCH_SIZE': 32,
    'BERT_BATCH_SIZE': 16,

    # 文件路径
    'GLOVE_PATH': os.path.join(os.getcwd(), "data", "glove.6B.300d.txt"),
    'SAVE_DIR': os.path.join(os.getcwd(), "saved_models"),
    'BERT_SAVE_DIR': os.path.join(os.getcwd(), "saved_models", "bert"),
    'TRAIN_FILE': 'imdb_train_processed.csv',
    'TEST_FILE': 'imdb_test_processed.csv',
    'VOCAB_FILE': os.path.join('data', 'vocab.txt'),
    'BERT_MAX_LENGTH': 512,

    # BERT 模型配置
    'BERT_MODEL_NAME': 'bert-base-uncased',  # 模型名称
    'BERT_CACHE_DIR': os.path.join(os.getcwd(), "bert_cache"),  # 缓存目录
    'BERT_LOCAL_PATH': os.path.join(os.getcwd(), "bert_cache", "bert-base-uncased")  # 本地模型路径
}
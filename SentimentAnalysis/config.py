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
    'EPOCHS': 10,
    'BERT_EPOCHS': 10,
    'BATCH_SIZE': 32,
    'BERT_BATCH_SIZE': 16,

    # 数据加载参数
    'NUM_WORKERS': 6,  # 根据您的硬件优化设置

    # 文件路径
    'GLOVE_PATH': os.path.join(os.getcwd(), "data", "glove.6B.300d.txt"),
    'SAVE_DIR': os.path.join(os.getcwd(), "saved_models"),
    'BERT_SAVE_DIR': os.path.join(os.getcwd(), "saved_models", "bert"),
    'TRAIN_FILE': 'word_level/train_data.csv',
    'TEST_FILE': 'word_level/test_data.csv',
    'VOCAB_FILE': os.path.join('data', 'vocab.txt'),
    'BERT_MAX_LENGTH': 512,

    # BERT 模型配置
    'BERT_MODEL_NAME': 'bert-base-uncased',  # 模型名称
    'BERT_CACHE_DIR': os.path.join(os.getcwd(), "bert_cache"),  # 缓存目录
    'BERT_LOCAL_PATH': os.path.join(os.getcwd(), "bert_cache", "bert-base-uncased")  # 本地模型路径
}

EXPERIMENT_CONFIG = {
    # LSTM实验参数
    'LSTM_EXPERIMENTS': {
        'embedding_types': ['random', 'glove_frozen', 'glove_unfrozen'],
        'hidden_dims': [128, 256, 512],
        'n_layers_list': [1, 2, 3],
        'dropouts': [0.3, 0.5, 0.7],
        'learning_rates': [1e-3, 5e-4, 1e-4],
        'batch_sizes': [32, 64, 128]
    },

    # BERT实验参数
    'BERT_EXPERIMENTS': {
        'strategies': ['full_finetune', 'last_2_layers', 'last_4_layers', 'classifier_only'],
        'learning_rates': [1e-5, 2e-5, 3e-5, 5e-5],
        'batch_sizes': [16, 32],
        'max_lengths': [128, 256, 512],
        'classifier_sizes': [128, 256, 512]
    },

    # 实验设置
    'EXPERIMENT_SETTINGS': {
        'lstm_epochs': 5,  # 实验阶段用较少epochs
        'bert_epochs': 3,
        'save_results': True,
        'results_dir': 'experiment_results'
    }
}
"""
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

    # 数据加载参数
    'NUM_WORKERS': 6,  # 根据您的硬件优化设置

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
}"""
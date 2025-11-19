import torch
import time
import os

from SentimentAnalysis import data_loader
from SentimentAnalysis.bert_model import BERTSentimentClassifier
from SentimentAnalysis.bert_train import BERTFineTuningStrategies, BERTTrainer
from SentimentAnalysis.IMDBDatasetBERT import IMDBDatasetBERT
from torch.utils.data import DataLoader


def create_bert_model(strategy='last_2_layers'):
    """创建BERT微调模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    strategy_config = BERTFineTuningStrategies.get_strategy(strategy)

    model = BERTSentimentClassifier(
        dropout=strategy_config['dropout'],
        freeze_bert_layers=strategy_config['freeze_bert_layers'],
        classifier_hidden_size=strategy_config['classifier_hidden_size']
    ).to(device)

    return model, strategy_config, device

def train_bert():
    """训练BERT微调模型"""
    print("=== BERT微调训练 ===")
    strategy = "last_2_layers"
    # 创建模型
    model, strategy_config ,device = create_bert_model(strategy)

    result = data_loader.load_bert_data()
    if result is None:
        print("数据加载失败!")
        return None
    train_loader, val_loader, test_loader = result

    # 创建训练器
    trainer = BERTTrainer(model, train_loader, val_loader, device)

    # 开始训练
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # 在测试集上评估
    test_metrics = trainer.evaluate(test_loader)
    print(f"\n=== Bert 策略： {strategy} 模型最终结果 ===")
    print(f"测试集准确率: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"测试集F1分数: {test_metrics['f1']:.3f}")

    return {
        'test_metrics': test_metrics,
        'training_time': training_time,
        'model': model,
        'trainer': trainer
    }

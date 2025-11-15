# 主训练流程
import torch
from SentimentAnalysis import data_loader
def main_training():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, val_loader, test_loader = data_loader.main()  # 调用之前的数据处理函数

    # 模型参数
    VOCAB_SIZE = 25000  # 根据你的词汇表大小调整
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT = 0.5
    PAD_IDX = 0  # padding token的索引

    # 创建模型（随机初始化嵌入）
    model = SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
        embedding_matrix=None,  # 随机初始化
        freeze_embedding=False
    )

    model = model.to(device)

    # 创建训练器
    trainer = SentimentTrainer(model, train_loader, val_loader, device)

    # 开始训练
    trainer.train(epochs=20)

    # 绘制训练历史
    trainer.plot_training_history()

    # 在测试集上评估
    print("\n=== 测试集评估 ===")
    test_metrics = trainer.evaluate(test_loader)
    print(f"测试集准确率: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"测试集F1分数: {test_metrics['f1']:.3f}")

    return trainer, test_metrics
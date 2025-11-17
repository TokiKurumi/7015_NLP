# 主训练流程
import torch
from SentimentAnalysis import data_loader,train
from SentimentAnalysis import model as m
import time
import os
from tqdm import tqdm

def train_model(embedding_type, freeze_embedding=False, glove_path=None):
    """
    训练指定嵌入类型的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== 训练 {embedding_type} 嵌入模型 ===")
    print(f"冻结嵌入层: {freeze_embedding}")
    print(f"使用设备: {device}")
    # 加载数据 - 修复返回值问题
    result = data_loader.main()
    if result is None:
        print("数据加载失败!")
        return None
    train_loader, val_loader, test_loader, vocab_size, vocab = result

    # 如果data_loader返回了vocab，使用它，否则需要从文件加载
    vocab = None
    if len(result) > 4:  # 如果返回了vocab
        vocab = result[4]
    else:
        # 从文件加载vocab
        vocab_file = os.path.join('data', 'vocab.txt')
        if os.path.exists(vocab_file):
            vocab = data_loader.load_vocabulary(vocab_file)
        else:
            print("警告: 无法加载词汇表")

    # 模型参数
    VOCAB_SIZE = vocab_size
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT = 0.5
    PAD_IDX = 0

    # 选择嵌入方式
    if embedding_type == "random":
        embedding_matrix = None
        print("使用随机初始化嵌入")
    elif embedding_type == "glove":
        if glove_path is None:
            print("错误: 使用GloVe需要提供glove_path")
            return None
        if vocab is None:
            print("错误: 无法加载词汇表，无法使用GloVe")
            return None
        embedding_matrix = data_loader.load_glove_embeddings(
            glove_path, vocab, EMBEDDING_DIM
        )
        if embedding_matrix is None:
            print("错误: 加载GloVe词向量失败")
            return None
        print("使用预训练GloVe词向量")
    else:
        print(f"不支持的嵌入类型: {embedding_type}")
        return None

    # 创建模型
    model = m.SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
        embedding_matrix=embedding_matrix,
        freeze_embedding=freeze_embedding
    ).to(device)

    # 为不同模型类型创建不同的保存目录
    model_suffix = embedding_type
    if embedding_type == "glove":
        model_suffix += "_frozen" if freeze_embedding else "_unfrozen"

    save_dir = f'saved_models/{model_suffix}'

    # 创建训练器
    trainer = train.SentimentTrainer(model, train_loader, val_loader, device, save_dir=save_dir)

    # 开始训练
    start_time = time.time()
    trainer.train(epochs=5)

    training_time = time.time() - start_time

    # 在测试集上评估
    test_metrics = trainer.evaluate(test_loader)
    print(f"\n=== {embedding_type} 模型最终结果 ===")
    print(f"测试集准确率: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"测试集F1分数: {test_metrics['f1']:.3f}")

    return {
        'embedding_type': embedding_type,
        'freeze_embedding': freeze_embedding,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'model': model,
        'trainer': trainer
    }


def compare_embeddings():
    """
    比较两种嵌入方式的效果
    """
    results = {}
    print("开始比较不同嵌入方式的性能...")
    # 训练随机初始化模型

    # 训练GloVe模型（不冻结）
    print("\n2. 训练GloVe模型（不冻结）...")
    results['glove_unfrozen'] = train_model("glove", freeze_embedding=False,
                                            glove_path="glove.6B.100d.txt")

    # 训练GloVe模型（冻结）
    print("\n3. 训练GloVe模型（冻结）...")
    results['glove_frozen'] = train_model("glove", freeze_embedding=True,
                                          glove_path="glove.6B.100d.txt")

    # 打印比较结果
    print("\n" + "=" * 60)
    print("嵌入方式比较结果")
    print("=" * 60)

    for name, result in results.items():
        if result is not None:
            metrics = result['test_metrics']
            print(f"\n{name}:")
            print(f"  准确率: {metrics['accuracy'] * 100:.2f}%")
            print(f"  F1分数: {metrics['f1']:.3f}")
            print(f"  精确率: {metrics['precision']:.3f}")
            print(f"  召回率: {metrics['recall']:.3f}")
            print(f"  训练时间: {result['training_time']:.2f}秒")
        else:
            print(f"\n{name}: 训练失败")

    return results

def main():
    print("选择训练模式:")
    print("1. 随机初始化嵌入")
    print("2. 预训练GloVe嵌入（不冻结）")
    print("3. 预训练GloVe嵌入（冻结）")
    print("4. 比较所有嵌入方式")

    choice = input("请输入选择 (1-4): ").strip()

    if choice == "1":
        result = train_model("random")
        if result is None:
            print("随机初始化模型训练失败!")
    elif choice == "2":
        result = train_model("glove", freeze_embedding=False, glove_path="glove.6B.100d.txt")
        if result is None:
            print("GloVe不冻结模型训练失败!")
    elif choice == "3":
        result = train_model("glove", freeze_embedding=True, glove_path="glove.6B.100d.txt")
        if result is None:
            print("GloVe冻结模型训练失败!")
    elif choice == "4":
        results = compare_embeddings()
        # 保存比较结果
        if any(results.values()):
            print(f"\n所有模型训练完成! 最佳模型保存在各自的saved_models子目录中")
    else:
        print("无效选择")

if __name__ == '__main__':
    main()
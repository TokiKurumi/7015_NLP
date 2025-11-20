# 主训练流程
import torch
from SentimentAnalysis import data_loader,train
from SentimentAnalysis import model as m
from SentimentAnalysis.bert_model import BERTSentimentClassifier
from SentimentAnalysis.bert_train import BERTFineTuningStrategies, BERTTrainer
import time
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
from config import CONFIG


def get_model_config(vocab_size):
    """获取模型配置"""
    return {
        'vocab_size': vocab_size,
        'embedding_dim': CONFIG['EMBEDDING_DIM'],
        'hidden_dim': CONFIG['HIDDEN_DIM'],
        'output_dim': CONFIG['OUTPUT_DIM'],
        'n_layers': CONFIG['N_LAYERS'],
        'dropout': CONFIG['DROPOUT'],
        'pad_idx': CONFIG['PAD_IDX']
    }


def create_model(model_config, embedding_type, vocab=None, freeze_embedding=False):
    """创建指定嵌入类型的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if embedding_type == "random":
        embedding_matrix = None
        print("使用随机初始化嵌入")
    elif embedding_type == "glove":
        if vocab is None:
            print("错误: 使用GloVe需要词汇表")
            return None

        embedding_matrix = data_loader.load_glove_embeddings(
            CONFIG['GLOVE_PATH'], vocab, CONFIG['EMBEDDING_DIM']
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
        **model_config,
        embedding_matrix=embedding_matrix,
        freeze_embedding=freeze_embedding
    ).to(device)

    return model, device

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
    model, strategy_config, device = create_bert_model(strategy)

    result = data_loader.load_bert_data(
        batch_size=CONFIG['BERT_BATCH_SIZE'],
        max_length=CONFIG['BERT_MAX_LENGTH']
    )
    if result is None:
        print("数据加载失败!")
        return None
    train_loader, val_loader, test_loader = result

    # 创建训练器
    trainer = BERTTrainer(model, train_loader, val_loader, device,
                         save_dir=CONFIG['BERT_SAVE_DIR'],
                         epochs=CONFIG['BERT_EPOCHS'])

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

def train_model(embedding_type, freeze_embedding=False):
    """
    训练指定嵌入类型的模型
    """
    print(f"\n=== 训练 {embedding_type} 嵌入模型 ===")
    print(f"冻结嵌入层: {freeze_embedding}")
    # 加载数据 - 修复返回值问题
    result = data_loader.load_data()
    if result is None:
        print("数据加载失败!")
        return None
    train_loader, val_loader, test_loader, vocab_size, vocab = result

    # 获取模型配置
    model_config = get_model_config(vocab_size)

    # 创建模型
    model, device = create_model(model_config, embedding_type, vocab, freeze_embedding)
    if model is None:
        return None

    print(f"使用设备: {device}")

    # 为不同模型类型创建不同的保存目录
    model_suffix = embedding_type
    if embedding_type == "glove":
        model_suffix += "_frozen" if freeze_embedding else "_unfrozen"

    save_dir = os.path.join(CONFIG['SAVE_DIR'], model_suffix)


    # 创建训练器
    trainer = train.SentimentTrainer(
        model, train_loader, val_loader, device,
        save_dir=save_dir,
        epochs=CONFIG['EPOCHS']
    )

    # 开始训练
    start_time = time.time()
    trainer.train()
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

    # 1. 训练随机初始化模型
    print("\n1. 训练随机初始化模型...")
    results['random'] = train_model("random")

    # 2. 训练GloVe模型（不冻结）
    print("\n2. 训练GloVe模型（不冻结）...")
    results['glove_unfrozen'] = train_model("glove", freeze_embedding=False)

    # 3. 训练GloVe模型（冻结）
    print("\n3. 训练GloVe模型（冻结）...")
    results['glove_frozen'] = train_model("glove", freeze_embedding=True)

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


def run_advanced_analysis():
    """运行高级数据分析"""
    print("=== 高级数据分析 ===")

    try:
        from my_data_analysis import IMDBDataAnalyzer

        # 创建分析器实例
        analyzer = IMDBDataAnalyzer("aclImdb")

        # 加载数据
        train_df, test_df = analyzer.load_imdb_from_local()
        if train_df is None:
            print("数据加载失败!")
            return

        # 文本清洗
        analyzer.apply_text_cleaning()

        # 选择分词类型
        print("\n请选择分词类型:")
        print("1. Word-level (单词级)")
        print("2. Character-level (字符级)")
        print("3. Subword-level (子词级)")

        choice = input("请输入选择 (1-3): ").strip()

        tokenization_type = 'word'
        tokenization_params = {}

        if choice == '1':
            tokenization_type = 'word'
        elif choice == '2':
            tokenization_type = 'char'
        elif choice == '3':
            tokenization_type = 'subword'
            try:
                min_n = int(input("请输入最小子词长度 (默认2): ") or 2)
                max_n = int(input("请输入最大子词长度 (默认4): ") or 4)
                tokenization_params = {'min_n': min_n, 'max_n': max_n}
            except ValueError:
                print("使用默认参数")
                tokenization_params = {'min_n': 2, 'max_n': 4}

        # 应用分词
        analyzer.apply_tokenization(tokenization_type, **tokenization_params)

        # 统计分析
        analyzer.calculate_basic_statistics()

        # 构建词汇表（仅单词级分词）
        if tokenization_type == 'word':
            analyzer.build_vocabulary()
            analyzer.convert_to_sequences(256)
            analyzer.pad_sequences()
            analyzer.split_train_val()

        # 可视化
        analyzer.create_visualizations('data_distribution.png')

        # 保存处理后的数据
        save_choice = input("是否保存处理后的数据? (y/n): ").lower().strip()
        if save_choice == 'y':
            analyzer.save_processed_data()

        print("\n数据分析完成!")

    except Exception as e:
        print(f"高级数据分析失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("选择训练模式:")
    print("1. 随机初始化嵌入")
    print("2. 预训练GloVe嵌入（不冻结）")
    print("3. 预训练GloVe嵌入（冻结）")
    print("4. 比较所有嵌入方式")
    print("5. 运行基础数据分析")
    print("6. 运行高级数据分析")  # 新增选项
    print("7. 运行BERT微调")

    choice = input("请输入选择 (1-7): ").strip()

    if choice == "1":
        result = train_model("random")
        if result is None:
            print("随机初始化模型训练失败!")
    elif choice == "2":
        result = train_model("glove", freeze_embedding=False)
        if result is None:
            print("GloVe不冻结模型训练失败!")
    elif choice == "3":
        result = train_model("glove", freeze_embedding=True)
        if result is None:
            print("GloVe冻结模型训练失败!")
    elif choice == "4":
        results = compare_embeddings()
    elif choice == "5":
        print("运行基础数据分析...")
        try:
            from my_data_analysis import run_my_analysis
            stats = run_my_analysis()
            print("\n数据分析统计结果:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"数据分析运行失败: {e}")
    elif choice == "6":  # 新增高级数据分析
        run_advanced_analysis()
    elif choice == "7":
        # 让用户选择BERT实验参数
        print("\n选择BERT实验配置:")
        print("1. 默认配置 (last_2_layers, lr=2e-5, batch=16, len=256)")
        print("2. 全参数微调 (full_finetune, lr=2e-5, batch=16, len=256)")
        print("3. 仅分类器 (classifier_only, lr=1e-4, batch=16, len=256)")
        print("4. 自定义配置")

        bert_choice = input("请输入选择 (1-4): ").strip()

        if bert_choice == "1":
            result = train_bert(strategy="last_2_layers", learning_rate=2e-5, batch_size=16, max_length=256)
        elif bert_choice == "2":
            result = train_bert(strategy="full_finetune", learning_rate=2e-5, batch_size=16, max_length=256)
        elif bert_choice == "3":
            result = train_bert(strategy="classifier_only", learning_rate=1e-4, batch_size=16, max_length=256)
        elif bert_choice == "4":
            # 自定义配置
            strategy = input("输入微调策略 (last_2_layers/full_finetune/classifier_only): ").strip()
            lr = float(input("输入学习率 (如2e-5): ").strip())
            batch_size = int(input("输入批次大小 (如16): ").strip())
            max_length = int(input("输入序列长度 (如256): ").strip())
            result = train_bert(strategy=strategy, learning_rate=lr, batch_size=batch_size, max_length=max_length)
        else:
            result = train_bert()  # 使用默认

        if result is None:
            print("BERT微调训练失败!")
    else:
        print("无效选择")


if __name__ == '__main__':
    main()

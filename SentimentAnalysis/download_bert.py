# download_bert.py
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 导入共享配置
from config import CONFIG


def download_bert_model():
    """下载BERT模型文件"""
    print("开始下载BERT模型文件...")

    # 创建缓存目录
    os.makedirs(CONFIG['BERT_CACHE_DIR'], exist_ok=True)

    model_name = CONFIG['BERT_MODEL_NAME']

    try:
        # 下载模型
        print("下载BERT模型...")
        model = AutoModel.from_pretrained(model_name, cache_dir=CONFIG['BERT_CACHE_DIR'])

        # 下载tokenizer
        print("下载BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CONFIG['BERT_CACHE_DIR'])

        # 下载config
        print("下载BERT config...")
        config = AutoConfig.from_pretrained(model_name, cache_dir=CONFIG['BERT_CACHE_DIR'])

        print("BERT模型下载完成！")
        print(f"文件保存在: {CONFIG['BERT_CACHE_DIR']}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        print("请检查网络连接")
        return False


if __name__ == "__main__":
    download_bert_model()
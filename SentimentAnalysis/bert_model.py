import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BertModel, BertConfig
from config import CONFIG
import os
class BERTSentimentClassifier(nn.Module):


    def __init__(self,  dropout=0.2,freeze_bert_layers=None, classifier_hidden_size=256):
        """
        :param freeze_bert_layers:  this param can be string = 'all' , int (freeze first n layers)
        :param classifier_hidden_size: set the mlp hidden size
        """
        super().__init__()
        local_path = CONFIG['BERT_LOCAL_PATH']
        try:
            # 检查本地是否有配置文件
            if os.path.exists(os.path.join(local_path, "bert_config.json")):
                print(f"从本地加载BERT模型: {local_path}")

                self.config = AutoConfig.from_pretrained(local_path)

                # 对于 TensorFlow 格式的检查点文件，我们需要特殊处理
                # 由于您提供的是 TensorFlow 格式，我们需要使用 PyTorch 版本的模型
                # 这里我们回退到从 Hugging Face 下载 PyTorch 版本
                print("检测到TensorFlow格式模型，使用PyTorch版本的BERT...")
                self.bert = AutoModel.from_pretrained(CONFIG['BERT_MODEL_NAME'])
            else:
                print("本地BERT配置文件不存在，从网络下载...")
                self.bert = AutoModel.from_pretrained(CONFIG['BERT_MODEL_NAME'])
                self.config = AutoConfig.from_pretrained(CONFIG['BERT_MODEL_NAME'])

        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            print("尝试从网络下载BERT模型...")
            self.bert = AutoModel.from_pretrained(CONFIG['BERT_MODEL_NAME'])
            self.config = AutoConfig.from_pretrained(CONFIG['BERT_MODEL_NAME'])

        # freeze layer
        if freeze_bert_layers is not None:
            self.freeze_layers(freeze_bert_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, classifier_hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, 1)
        )

        self.init_weight()

        print(f"BERT微调模型初始化完成")
        if freeze_bert_layers:
            print(f"冻结层: {freeze_bert_layers}")


    def freeze_layers(self, freeze_bert_layers):
        if freeze_bert_layers == 'all':
            # 冻结所有BERT参数
            for param in self.bert.parameters():
                param.requires_grad = False
        elif isinstance(freeze_bert_layers, list):
            # 冻结指定层
            for layer_idx in freeze_bert_layers:
                if layer_idx < len(self.bert.encoder.layer):
                    for param in self.bert.encoder.layer[layer_idx].parameters():
                        param.requires_grad = False
        elif isinstance(freeze_bert_layers, int):
            for layer_idx in range(freeze_bert_layers):
                if layer_idx < len(self.bert.encoder.layer):
                    for param in self.bert.encoder.layer[layer_idx].parameters():
                        param.requires_grad = False

    def init_weight(self):
        # bert在预训练时使用的是正态分布初始化
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token通过自注意力机制与序列中所有其他token交互，因此它包含了整个序列的聚合信息
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)
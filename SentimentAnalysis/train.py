# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt


class SentimentTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 损失函数和优化器
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters())

        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0

        for batch in self.train_loader:
            sequences = batch['sequences'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(sequences, lengths).squeeze(1)
            loss = self.criterion(predictions, labels)

            # 计算准确率
            predicted_labels = torch.sigmoid(predictions) > 0.5
            acc = (predicted_labels == labels).float().mean()

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                sequences = batch['sequences'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                predictions = self.model(sequences, lengths).squeeze(1)
                loss = self.criterion(predictions, labels)

                predicted_probs = torch.sigmoid(predictions)
                predicted_labels = predicted_probs > 0.5
                acc = (predicted_labels == labels).float().mean()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算详细指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        return {
            'loss': epoch_loss / len(loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, epochs, early_stopping_patience=5):
        best_val_loss = float('inf')
        patience_counter = 0

        print("开始训练...")
        for epoch in range(epochs):
            start_time = time.time()

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_metrics = self.evaluate(self.val_loader)

            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])

            # 打印进度
            epoch_time = time.time() - start_time
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_time:.2f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\tVal Loss: {val_metrics["loss"]:.3f} | Val Acc: {val_metrics["accuracy"] * 100:.2f}%')
            print(
                f'\tVal Precision: {val_metrics["precision"]:.3f} | Val Recall: {val_metrics["recall"]:.3f} | Val F1: {val_metrics["f1"]:.3f}')

            # 早停
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"早停在第 {epoch + 1} 轮")
                    break

    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
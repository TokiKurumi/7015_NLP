# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import os


class SentimentTrainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir=None, epochs=10, learning_rate=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir if save_dir else os.path.join('saved_models')
        self.epochs = epochs

        os.makedirs(self.save_dir, exist_ok=True)

        # 损失函数和优化器
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epoch_times = []

        #Conserve the best model
        self.best_val_accuracy = 0.0
        self.best_model_path = os.path.join(save_dir, 'best_model.pt')

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0

        total_batches = len(self.train_loader)

        # 使用简单的百分比显示
        for i, batch in enumerate(self.train_loader):
            sequences = batch['sequences'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(sequences, lengths)

            if predictions.dim() > 1 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)

            loss = self.criterion(predictions, labels)

            # 计算准确率
            predicted_labels = torch.sigmoid(predictions) > 0.5
            acc = (predicted_labels == labels).float().mean()

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # 每10个批次或最后一个批次显示进度
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                progress = (i + 1) / total_batches * 100
                # 使用白色ANSI代码
                print(
                    f'\r\033[97mEpoch {epoch + 1}/{total_epochs}: Progress: {progress:6.2f}% | Loss: {loss.item():.4f} | Acc: {acc.item() * 100:6.2f}%\033[0m',
                    end='', flush=True)

        # 清除进度行
        print('\r\033[K', end='', flush=True)

        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        all_predictions = []
        all_labels = []

        total_batches = len(loader)
        with torch.no_grad():
            # 使用简单的百分比显示而不是进度条
            for i, batch in enumerate(loader):
                sequences = batch['sequences'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                predictions = self.model(sequences, lengths)
                if predictions.dim() > 1 and predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)
                loss = self.criterion(predictions, labels)

                predicted_probs = torch.sigmoid(predictions)
                predicted_labels = predicted_probs > 0.5
                acc = (predicted_labels == labels).float().mean()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 每10个批次或最后一个批次显示进度
                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    progress = (i + 1) / total_batches * 100
                    print(
                        f'\r\033[97m验证中: {progress:6.2f}% | Loss: {loss.item():.4f} | Acc: {acc.item() * 100:6.2f}%\033[0m',
                        end='', flush=True)

        # 清除进度行
        print('\r\033[K', end='', flush=True)

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

    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'best_val_accuracy': self.best_val_accuracy
            }

            torch.save(checkpoint, self.best_model_path)
            print(f"保存最佳模型到: {self.best_model_path} (epoch {epoch}, 准确率: {self.best_val_accuracy * 100:.2f}%)")





    def train(self):

        print("开始训练...")
        print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Val F1':^8} | {'Time(s)':^8} | {'Best':^6}")
        print("-" * 95)

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch, self.epochs)

            # 验证
            val_metrics = self.evaluate(self.val_loader)

            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])

            # 检查是否为最佳模型
            is_best = False
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                is_best = True
                self.save_checkpoint(epoch + 1, is_best=True)

            # 打印输出行
            print(f"{epoch + 1:>6} | {train_loss:>10.4f} | {train_acc * 100:>9.2f}% | "
                  f"{val_metrics['loss']:>10.4f} | {val_metrics['accuracy'] * 100:>9.2f}% | "
                  f"{val_metrics['f1']:>8.3f} | {epoch_time:>8.2f} | {'√' if is_best else '×':^6}")

        self.plot_training_history()

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        total_training_time = sum(self.epoch_times)
        print(f"\n训练完成!")
        print(f"总训练时间: {total_training_time:.2f}秒")
        print(f"平均每个epoch时间: {avg_epoch_time:.2f}秒")
        print(f"最佳验证集准确率: {self.best_val_accuracy * 100:.2f}%")
        print(f"最佳模型保存在: {self.best_model_path}")

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

    def load_best_model(self):
        """加载最佳模型"""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"已加载最佳模型 (epoch {checkpoint['epoch']}, 准确率: {checkpoint['best_val_accuracy'] * 100:.2f}%)")
            return True
        else:
            print("最佳模型文件不存在")
            return False
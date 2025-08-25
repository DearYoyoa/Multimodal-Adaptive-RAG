import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 数据集定义
class TokenProbeDataset(Dataset):
    def __init__(self, json_file, image_hidden_state_dir, original_image_hidden_state_dir, layer_idx=-1):
        """
        Args:
            layer_idx (int): 指定使用第几层的hidden state。默认为-1，表示最后一层。
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_hidden_state_dir = image_hidden_state_dir
        self.original_image_hidden_state_dir = original_image_hidden_state_dir
        self.layer_idx = layer_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['image_id']

        # 加载 image hidden states (33, 2, 4096)
        image_hidden_state = np.load(os.path.join(self.image_hidden_state_dir, f"{image_id}.npy"))

        # 加载 original image hidden states (2, 4096)
        original_image_hidden_state = np.load(os.path.join(self.original_image_hidden_state_dir, f"{image_id}.npy"))

        # 选择指定层的 hidden state
        selected_hidden_state = image_hidden_state[self.layer_idx]  # (2, 4096)

        # 获取标签
        label = 0 if item['group'].endswith('_0') else 1

        return (
            torch.tensor(selected_hidden_state, dtype=torch.float32),  # (2, 4096)
            torch.tensor(original_image_hidden_state, dtype=torch.float32),  # (2, 4096)
            torch.tensor(label, dtype=torch.long)
        )

# 二分类器定义
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.fc(x)

# 训练模型并保存
def train_model(model, dataloader, criterion, optimizer, device, model_name, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for image_hidden, original_hidden, labels in dataloader:
            inputs = image_hidden[:, 0, :]  # 选择 (2, 4096) 中的第一个 token
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 保存模型
        save_model(model, model_name, epoch+1)

# 评估模型
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for image_hidden, original_hidden, labels in dataloader:
            inputs = image_hidden[:, 0, :].to(device)  # 选择第一个 token
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    logging.info(f"Test Accuracy: {accuracy:.4f}")

# 保存模型
def save_model(model, model_name, epoch):
    save_dir = f'models/token_probe_evqa_i2_test/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved model to {save_path}")


def main():
    # 超参数配置
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    weight_decay = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 选择层索引
    layer_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -1]

    # 训练每个层的模型
    for layer_idx in layer_indices:
        logging.info(f"\nTraining binary classification model for layer {layer_idx}")

        dataset = TokenProbeDataset(
            json_file="evqa_train_data/evqa/evqa_train_i2/logs_evqa_train_i2.json",
            image_hidden_state_dir="data_image_token_i2_train/data_image_token_evqa/hidden_state",
            original_image_hidden_state_dir="data_image_token_i2_train/data_image_token_evqa/original_hidden_state",
            layer_idx=layer_idx
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model_name = f"qwen_dp3_binary_layer_{layer_idx}_image_wo_rir"
        model = BinaryClassifier(input_dim=4096, num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_model(model, dataloader, criterion, optimizer, device, model_name, num_epochs)
        evaluate_model(model, dataloader, device)

if __name__ == "__main__":
    main() 

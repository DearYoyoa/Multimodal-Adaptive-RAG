import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
from tqdm import tqdm
import time

class TokenProbeDataset(Dataset):
    def __init__(self, json_file, image_hidden_state_dir, original_image_hidden_state_dir, 
                 hidden_state_dir=None, hidden_state_rir_dir=None, layer_idx=-1):
        """
        Args:
            layer_idx (int): 指定使用第几层的hidden state。默认为-1，表示最后一层。
                           范围应该在0到32之间（总共33层，索引从0开始）
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_hidden_state_dir = image_hidden_state_dir
        self.original_image_hidden_state_dir = original_image_hidden_state_dir
        self.hidden_state_dir = hidden_state_dir
        self.hidden_state_rir_dir = hidden_state_rir_dir
        self.layer_idx = layer_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['image_id']
        
        # 加载image hidden states - (33, 2, 4096)
        image_hidden_state = np.load(os.path.join(self.image_hidden_state_dir, f"{image_id}.npy"))
        # 加载original image hidden states - (2, 4096)
        original_image_hidden_state = np.load(os.path.join(self.original_image_hidden_state_dir, f"{image_id}.npy"))
        
        # 加载text hidden states
        hidden_state = None
        if self.hidden_state_dir:
            hidden_state = np.load(os.path.join(self.hidden_state_dir, f"{image_id}.npy"))
        
        hidden_state_rir = None
        if self.hidden_state_rir_dir:
            hidden_state_rir = np.load(os.path.join(self.hidden_state_rir_dir, f"{image_id}.npy"))
        
        # 获取标签
        label = 1 if item['group'].endswith('_0') else 0
        group_label = int(item['group'].split('_')[0]) * 2 + (1 if item['group'].endswith('_1') else 0)
        
        return (torch.tensor(image_hidden_state, dtype=torch.float32),  # (33, 2, 4096)
                torch.tensor(original_image_hidden_state, dtype=torch.float32),  # (2, 4096)
                torch.tensor(hidden_state, dtype=torch.float32) if hidden_state is not None else None,
                torch.tensor(hidden_state_rir, dtype=torch.float32) if hidden_state_rir is not None else None,
                torch.tensor(label, dtype=torch.long),
                torch.tensor(group_label, dtype=torch.long))

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
    
        return self.fc(x)

def train(model, dataloader, criterion, optimizer, device, epoch, model_type, num_classes, layer_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - {model_type}', leave=False)
    for batch in pbar:
        image_hidden, original_image_hidden, hidden_state, hidden_state_rir, label, group_label = [
            item.to(device) if item is not None else None for item in batch
        ]
        
        optimizer.zero_grad()
        
        # 根据模型类型选择输入
        if 'original' in model_type:
            if 'wo_rir' in model_type:
                # 只使用第一个图像的表征
                x = original_image_hidden[:, 0]  # (batch_size, 4096)
                # x = original_image_hidden
            else:
                # 使用所有表征
                x = original_image_hidden.reshape(original_image_hidden.size(0), -1)  # (batch_size, 8192)
        else:
            if 'wo_rir' in model_type:
                # 只使用指定层第一个图像的表征
                x = image_hidden[:, layer_idx, 0]  # (batch_size, 4096)
            else:
                # 使用指定层所有表征
                x = image_hidden[:, layer_idx].reshape(image_hidden.size(0), -1)  # (batch_size, 8192)
        
        # 选择文本表征
        if 'wo_rir' in model_type:
            text_hidden = hidden_state
        else:
            # 使用两种文本表征
            text_hidden = torch.cat([
                hidden_state.squeeze(1).squeeze(1),  # (batch_size, 4096)
                hidden_state_rir.squeeze(1).squeeze(1)  # (batch_size, 4096)
            ], dim=1)  # (batch_size, 8192)
            
        # 合并特征
        if text_hidden is not None:
            if 'wo_rir' in model_type:
                x = torch.cat((x, text_hidden.squeeze(1).squeeze(1)), dim=1)  # (batch_size, 8192)
            else:
                x = torch.cat((x, text_hidden), dim=1)  # (batch_size, 16384)

        outputs = model(x)
        target = label if num_classes == 2 else group_label
        
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    logging.info(f'Epoch {epoch} - {model_type} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def save_model(model, model_name, epoch):
    save_dir = f'models/token_probe_evqa_i3_prompt/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved model to {save_path}")

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 设置参数
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    weight_decay = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 为每一层创建数据集和训练模型
    layer_indices = [-1]  # 示例：选择几个关键层进行实验
    # 训练两种分类任务
    # for num_classes in [2, 4]:
    for num_classes in [4]:
        logging.info(f"\nTraining {num_classes}-class classification models")
        
        for layer_idx in layer_indices:
            logging.info(f"\nTraining models for layer {layer_idx}")
            
            # 创建数据集
            dataset = TokenProbeDataset(
                # json_file='data_0921/infoseek/infoseek_i2_rir/logs_infoseek_i2_rir.json',
                # json_file='okvqa_train_data/okvqa_i3_prompt/okvqa_train_i3_rir/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_okvqa_train_i3_rir.json',
                # image_hidden_state_dir='hidden_state_i3_train/hidden_okvqa_2/hidden_state',
                # original_image_hidden_state_dir='hidden_state_i3_train/hidden_okvqa_2/original_hidden_state',
                # hidden_state_dir='hidden_state_i3_train/hidden_okvqa_2/last_hidden_state',
                # hidden_state_rir_dir='hidden_state_i3_train/hidden_okvqa_2/last_hidden_state_rir',
                # layer_idx=layer_idx

                json_file='evqa_train_data/evqa_i3_prompt/evqa_train_i3_rir/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_evqa_train_i3_rir.json',
                image_hidden_state_dir='hidden_state_i3_train/hidden_evqa/hidden_state',
                original_image_hidden_state_dir='hidden_state_i3_train/hidden_evqa/original_hidden_state',
                hidden_state_dir='hidden_state_i3_train/hidden_evqa/last_hidden_state',
                hidden_state_rir_dir='hidden_state_i3_train/hidden_evqa/last_hidden_state_rir',
                layer_idx=layer_idx
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 定义模型配置
            task_prefix = 'binary' if num_classes == 2 else 'multiclass'
            # task_prefix = 'multiclass'
            model_configs = [
                # 不使用RIR的模型 (输入维度 4096 + 4096 = 8192)
                # (f'qwen_dp3_{task_prefix}_layer_{layer_idx}_original_image_rir', 7168, num_classes),
                # (f'qwen_dp3_{task_prefix}_layer_{layer_idx}_image_wo_rir', 4096, num_classes),
                # (f'qwen_dp3_{task_prefix}_layer_{layer_idx}_image_wo_rir', 8192, num_classes),
                # 使用所有表征的模型 (输入维度 8192 + 8192 = 16384)
                # (f'qwen_dp3_{task_prefix}_layer_{layer_idx}_original_image_rir', 16384, num_classes),
                (f'qwen_dp3_{task_prefix}_layer_{layer_idx}_image_rir', 16384, num_classes)
            ]

            # 训练每个模型
            for model_name, input_dim, num_classes in model_configs:
                logging.info(f"\nTraining {model_name}...")
                model = MLPClassifier(input_dim, num_classes).to(device)
                
                # 设置损失函数和优化器
                if num_classes == 2:
                    criterion = nn.CrossEntropyLoss()
                else:
                    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 40.0, 10.0, 5.0]).to(device))   # evqa best
                    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0, 1.0]).to(device)) # okvqa
                    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0, 40.0]).to(device))
                
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                for epoch in range(num_epochs):
                    start_time = time.time()
                    loss, accuracy = train(model, dataloader, criterion, optimizer, device, epoch+1, model_name, num_classes, layer_idx)
                    end_time = time.time()
                    logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {end_time-start_time:.2f} seconds")
                    save_model(model, model_name, epoch+1)

    logging.info("Training completed. All models saved.")

if __name__ == "__main__":
    main() 


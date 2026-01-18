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
    def __init__(self, json_file, hidden_state_dir=None, hidden_state_rir_dir=None):
        """
        Dataset class for Ablation experiments using only text hidden states.
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.hidden_state_dir = hidden_state_dir
        self.hidden_state_rir_dir = hidden_state_rir_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load text hidden states
        hidden_state = None
        if self.hidden_state_dir:
            hidden_state = np.load(os.path.join(self.hidden_state_dir, f"hidden_states_{item['image_id']}.npy"))
        
        hidden_state_rir = None
        if self.hidden_state_rir_dir:
            hidden_state_rir = np.load(os.path.join(self.hidden_state_rir_dir, f"hidden_states_rir_{item['image_id']}.npy"))
        
        # Get labels
        label = 0 if item['group'].endswith('_0') else 1
        group_label = int(item['group'].split('_')[0]) * 2 + (1 if item['group'].endswith('_1') else 0)
        
        return (
            torch.tensor(hidden_state, dtype=torch.float32) if hidden_state is not None else None,
            torch.tensor(hidden_state_rir, dtype=torch.float32) if hidden_state_rir is not None else None,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(group_label, dtype=torch.long)
        )

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

def train(model, dataloader, criterion, optimizer, device, epoch, model_type, num_classes):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - {model_type}', leave=False)
    for batch in pbar:
        hidden_state, hidden_state_rir, label, group_label = [
            item.to(device) if item is not None else None for item in batch
        ]
        
        optimizer.zero_grad()

        if 'wo_rir' in model_type:
            text_hidden = hidden_state.squeeze(1).squeeze(1)  # (batch_size, 4096)
        else:
            text_hidden = torch.cat([
                hidden_state.squeeze(1).squeeze(1),  # (batch_size, 4096)
                hidden_state_rir.squeeze(1).squeeze(1)  # (batch_size, 4096)
            ], dim=1)  # (batch_size, 8192)
        
        outputs = model(text_hidden)
        # outputs = outputs.view(-1, num_classes)  # (batch_size, num_classes)
        target = label if num_classes == 2 else group_label
        # print(f"outputs.shape: {outputs.shape}, target.shape: {target.shape}")
        
        
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
    save_dir = f'models/token_probe_evqa_i2_ablation_text/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved model to {save_path}")

def main():
    # Set logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    weight_decay = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train models for 2-class and 4-class tasks
    # for num_classes in [2, 4]:
    for num_classes in [4]:
        logging.info(f"\nTraining {num_classes}-class classification models")
        
        # Create dataset
        dataset = TokenProbeDataset(
            # json_file='data_0921/infoseek/infoseek_i2/logs_infoseek_i2.json',
            # hidden_state_dir='data_0921/infoseek/last_hidden_state',
            # hidden_state_rir_dir='data_0921/infoseek/last_hidden_state_rir'
            json_file='evqa_train_data/evqa/evqa_train_i2_rir/logs_evqa_train_i2_rir.json',
            hidden_state_dir='evqa_train_data/evqa/last_hidden_state',
            hidden_state_rir_dir='evqa_train_data/evqa/last_hidden_state_rir'
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define model configurations
        task_prefix = 'binary' if num_classes == 2 else 'multiclass'
        model_configs = [
            # Without RIR (Input dimension 4096)
            (f'qwen_dp3_{task_prefix}_text_wo_rir', 4096, num_classes),
            # With RIR (Input dimension 8192)
            (f'qwen_dp3_{task_prefix}_text_rir', 8192, num_classes)
        ]

        # Train each model
        for model_name, input_dim, num_classes in model_configs:
            logging.info(f"\nTraining {model_name}...")
            model = MLPClassifier(input_dim, num_classes).to(device)
            
            # Set loss function and optimizer
            if num_classes == 2:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0, 10.0]).to(device))
            
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            for epoch in range(num_epochs):
                start_time = time.time()
                loss, accuracy = train(model, dataloader, criterion, optimizer, device, epoch+1, model_name, num_classes)
                end_time = time.time()
                logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {end_time-start_time:.2f} seconds")
                save_model(model, model_name, epoch+1)

    logging.info("Training completed. All models saved.")

if __name__ == "__main__":
    main()

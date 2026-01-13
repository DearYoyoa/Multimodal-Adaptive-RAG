import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
import logging
from tqdm import tqdm
import time

class InfoseekDataset(Dataset):
    def __init__(self, json_file, image_dir, screenshot_dir=None, hidden_state_dir=None, hidden_state_rir_dir=None, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.screenshot_dir = screenshot_dir
        self.hidden_state_dir = hidden_state_dir
        self.hidden_state_rir_dir = hidden_state_rir_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['image_id']
        
        image_path = os.path.join(self.image_dir, f"{image_id}.JPEG")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        screenshot = None
        if self.screenshot_dir:
            screenshot_path = os.path.join(self.screenshot_dir, f"{image_id}-search_result.png")
            screenshot = Image.open(screenshot_path).convert('RGB')
            if self.transform:
                screenshot = self.transform(screenshot)
        
        hidden_state = None
        if self.hidden_state_dir:
            hidden_state_path = os.path.join(self.hidden_state_dir, f"hidden_states_{image_id}.npy")
            hidden_state = np.load(hidden_state_path)
        
        hidden_state_rir = None
        if self.hidden_state_rir_dir:
            hidden_state_rir_path = os.path.join(self.hidden_state_rir_dir, f"hidden_states_rir_{image_id}.npy")
            hidden_state_rir = np.load(hidden_state_rir_path)
        
        label = 1 if item['group'].endswith('_0') else 0
        group_label = int(item['group'].split('_')[0]) * 2 + (1 if item['group'].endswith('_1') else 0)
        
        return image, screenshot, hidden_state, hidden_state_rir, label, group_label

class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(4608, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, image, hidden_state):
        image_features = self.resnet(image)
        hidden_state = hidden_state.squeeze(1).squeeze(1)
        combined_features = torch.cat((image_features, hidden_state), dim=1)
        return self.fc(combined_features)

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  
        self.fc = nn.Sequential(
            nn.Linear(4608 * 2, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, image, screenshot, hidden_state, hidden_state_rir):
        image_features = self.resnet(image)
        screenshot_features = self.resnet(screenshot)
        hidden_state = hidden_state.squeeze(1).squeeze(1)
        hidden_state_rir = hidden_state_rir.squeeze(1).squeeze(1)
        combined_features = torch.cat((image_features, screenshot_features, hidden_state, hidden_state_rir), dim=1)
        return self.fc(combined_features)

def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        image, screenshot, hidden_state, hidden_state_rir, label, group_label = [item.to(device) if item is not None else None for item in batch]
        
        optimizer.zero_grad()
        
        if isinstance(model, Classifier1):
            outputs = model(image, hidden_state)
            target = label
        else:
            outputs = model(image, screenshot, hidden_state, hidden_state_rir)
            target = group_label
        
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
    logging.info(f'Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def save_model(model, model_name, epoch):
    save_dir = f'models/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved model to {save_path}")

def save_model(model, model_name, epoch):
    save_dir = f'models/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved model to {save_path}")

def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.0001
    weight_decay = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = InfoseekDataset(
        json_file='data_0921/infoseek/infoseek_i2_rir/logs_infoseek_i2_rir.json',
        image_dir='data_0921/image',
        screenshot_dir='data_0921/screenshot',
        hidden_state_dir='data_0921/infoseek/last_hidden_state',
        hidden_state_rir_dir='data_0921/infoseek/last_hidden_state_rir',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    classifier1 = Classifier1().to(device)
    classifier2 = Classifier2().to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0, 10.0]).to(device))
    optimizer1 = optim.AdamW(classifier1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer2 = optim.AdamW(classifier2.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # logging.info("Training Classifier 1...")
    # for epoch in range(num_epochs):
    #     start_time = time.time()
    #     loss, accuracy = train(classifier1, dataloader, criterion1, optimizer1, device, epoch+1)
    #     end_time = time.time()
    #     logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {end_time-start_time:.2f} seconds")
    #     save_model(classifier1, 'classifier1', epoch+1)

    logging.info("\nTraining Classifier 2...")
    for epoch in range(num_epochs):
        start_time = time.time()
        loss, accuracy = train(classifier2, dataloader, criterion2, optimizer2, device, epoch+1)
        end_time = time.time()
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {end_time-start_time:.2f} seconds")
        save_model(classifier2, 'classifier2', epoch+1)

    logging.info("Training completed. All models saved.")

if __name__ == "__main__":
    main()



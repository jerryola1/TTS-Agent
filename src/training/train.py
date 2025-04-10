import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import os
from pathlib import Path
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Config

class VoiceDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.data_dir = Path(config['data']['processed_data_dir'])
        self.audio_files = list(self.data_dir.glob('*.wav'))
        
    def __len__(self):
        return len(self.audio_files)
        
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(audio_path, sr=self.config['model']['sample_rate'])
        
        # Ensure audio is within length limits
        max_length = self.config['model']['sample_rate'] * self.config['model']['max_audio_length']
        if len(audio) > max_length:
            audio = audio[:max_length]
            
        return torch.FloatTensor(audio)

class VoiceCloningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize base model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config['model']['base_model'])
        
        # Add speaker embedding layer
        self.speaker_embedding = nn.Embedding(1, 768)  # Single speaker for now
        
        # Add projection layers
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
    def forward(self, x):
        # Get wav2vec2 features
        features = self.wav2vec2(x).last_hidden_state
        
        # Get speaker embedding
        speaker_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        speaker_emb = self.speaker_embedding(speaker_id)
        
        # Combine features and speaker embedding
        combined = features + speaker_emb.unsqueeze(1)
        
        # Project to final representation
        output = self.projection(combined)
        return output

def train(config):
    # Set device
    device = torch.device(config['hardware']['device'])
    
    # Initialize model
    model = VoiceCloningModel(config).to(device)
    
    # Initialize dataset and dataloader
    train_dataset = VoiceDataset(config, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers']
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Training loop
    model.train()
    for epoch in range(config['training']['num_epochs']):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Calculate loss (placeholder for now)
            loss = torch.mean(output)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Loss: {total_loss/len(train_loader)}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_steps'] == 0:
            checkpoint_path = Path(config['models']['trained']) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(train_loader),
            }, checkpoint_path)

if __name__ == "__main__":
    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # Create output directory
    os.makedirs(config['models']['trained'], exist_ok=True)
    
    # Start training
    train(config) 
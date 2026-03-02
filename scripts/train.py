import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import sys
import os

# Ensure the script can import from its own directory when run directly in Colab
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import FairFaceVGG16
    from dataset import FairFaceDataset, get_transform
except ModuleNotFoundError:
    from Profiler.model.model import FairFaceVGG16
    from Profiler.model.dataset import FairFaceDataset, get_transform

# Check for GPU (CUDA or MPS/M1 Mac)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def train_model(data_dir, batch_size=32, epochs=5, lr=0.001, model_path="fairface_multi_task_model.pth"):
    print(f"Using device: {device}")

    # 1. Dataset & DataLoaders
    train_csv = f"{data_dir}/fairface_label_train.csv"
    val_csv = f"{data_dir}/fairface_label_val.csv"
    
    train_dataset = FairFaceDataset(csv_file=train_csv, root_dir=data_dir, transform=get_transform(is_train=True))
    val_dataset = FairFaceDataset(csv_file=val_csv, root_dir=data_dir, transform=get_transform(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model
    model = FairFaceVGG16(num_races=7)
    model = model.to(device)

    # 3. Loss Functions & Optimizer
    # Notice: the model freezes VGG backbone and trains the new layers ONLY.
    criterion_gender = nn.BCELoss()           # Binary cross-entropy
    criterion_age = nn.L1Loss()               # Mean absolute error (MAE)
    criterion_race = nn.CrossEntropyLoss()    # Multi-class cross-entropy

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # tqdm for a progress bar that looks nice
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets_gender, targets_age, targets_race in progress_bar:
            # Move everything to GPU
            images = images.to(device)
            targets_gender = targets_gender.to(device)
            targets_age = targets_age.to(device)
            targets_race = targets_race.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            preds_gender, preds_age, preds_race = model(images)
            
            # Calculate respective losses
            loss_gen = criterion_gender(preds_gender, targets_gender)
            loss_age = criterion_age(preds_age, targets_age)
            loss_race = criterion_race(preds_race, targets_race)
            
            # Combine losses
            loss = loss_gen + loss_age + loss_race
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            # Track training metrics
            train_loss += loss.item()
            progress_bar.set_postfix({'TotLoss': loss.item(), 'GenL': loss_gen.item(), 'AgeL': loss_age.item(), 'RaceL': loss_race.item()})
            
        print(f"Epoch {epoch+1} Average Training Loss: {train_loss / len(train_loader):.4f}")

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, targets_gender, targets_age, targets_race in progress_bar:
                images = images.to(device)
                targets_gender = targets_gender.to(device)
                targets_age = targets_age.to(device)
                targets_race = targets_race.to(device)
                
                preds_gender, preds_age, preds_race = model(images)
                
                loss_gen = criterion_gender(preds_gender, targets_gender)
                loss_age = criterion_age(preds_age, targets_age)
                loss_race = criterion_race(preds_race, targets_race)
                
                loss = loss_gen + loss_age + loss_race
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1} Average Validation Loss: {val_loss / len(val_loader):.4f}")
        print("-" * 50)
        
    print("Training Complete!")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to '{model_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FairFace Multi-Task Model")
    parser.add_argument("--data_dir", type=str, default="../FairFace", help="Path to FairFace dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-path", type=str, default="fairface_multi_task_model.pth", help="Path to save the trained model")
    
    args = parser.parse_args()
    
    # Specify the path where your fairface files are located (train/, val/, and the two csvs)
    # The user has "FairFace" folder in current directory with contents
    train_model(data_dir=args.data_dir, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, model_path=args.model_path)
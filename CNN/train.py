import os
import gc
import time 
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import CNN
from torchvision import models
from torch.cuda.amp import autocast

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Save checkpoint
def save_checkpoint(state, filename="./cnn_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

def tuning_model(train_loader, val_loader, IMG_SIZE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training loop ---
    print("Starting training...")
    start_time = time.time()
    start_epoch = 0
    best_val_acc = 0
    checkpoint_file = "./resnet_checkpoint.pth.tar"

    if os.path.exists(checkpoint_file):
        print("📦 Checkpoint found. Loading saved model...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"🔁 Returning from epoch {start_epoch} with best_val_acc = {best_val_acc:.2f}%")

    num_epochs = 500
    save_every = 3
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, verbose=True, mode='max')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = max(best_val_acc, val_acc)
            torch.save(model.state_dict(), "./resnet_best_model.pt")

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc
            }
            save_checkpoint(checkpoint)

        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping in main training loop")
            break
            
    # End the timer
    end_time = time.time()
    duration = end_time - start_time
    
    print("Training complete.")
    print(f"Total training time: {duration:.2f} seconds")
    
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    return model

def train_model(train_loader, val_loader, img_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    params = {
        "n_hidden_1": 64,
        "n_hidden_2": 128,
        "convkernel": 3,
        "poolkernel": 2,
        "num_of_layers": 4,
        "num_of_neurons": 52,
        "learning_rate": 1e-3
    }
    print("Starting training...")
    # Start the timer
    start_time = time.time()
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Define the model using the fixed hyperparameters
    model = CNN(
        n_hidden_1=params['n_hidden_1'],
        n_hidden_2=params['n_hidden_2'],
        convkernel=params['convkernel'],
        poolkernel=params['poolkernel'],
        num_of_layers=params['num_of_layers'],
        num_of_neurons=params['num_of_neurons'],
        img_size=img_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # --- Training loop ---
    start_epoch = 0
    best_val_acc = 0
    checkpoint_file = "./cnn_checkpoint.pth.tar"

    if os.path.exists(checkpoint_file):
        print("📦 Checkpoint found. Loading saved model...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"🔁 Returning from epoch {start_epoch} with best_val_acc = {best_val_acc:.2f}%")

    num_epochs = 500
    save_every = 3
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, verbose=True, mode='max')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = max(best_val_acc, val_acc)
            torch.save(model.state_dict(), "./cnn_best_model.pt")

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc
            }
            save_checkpoint(checkpoint)

        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping in main training loop")
            break
            
    # End the timer
    end_time = time.time()
    duration = end_time - start_time
    
    print("Training complete.")
    print(f"Total training time: {duration:.2f} seconds")
    
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    return model
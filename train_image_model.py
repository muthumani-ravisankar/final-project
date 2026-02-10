import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ADD THIS IMPORT
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split  # ADD THIS IMPORT
from tqdm import tqdm
import pandas as pd
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    DATA_PATH = r"C:\Users\padma\Downloads\archive (5)\real_vs_fake\real-vs-fake"
    MODEL_SAVE_PATH = "models"
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    IMG_SIZE = 224
    NUM_CLASSES = 2
    MODEL_NAME = 'efficientnet_b3'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 10
    USE_AMP = True  # Automatic Mixed Precision for faster training

config = Config()

# Create model directory
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

# Custom Dataset for manual train/val split
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label

# Advanced Image Dataset with Augmentation
class AIImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        # Define class mapping
        self.class_to_idx = {'real': 0, 'fake': 1}
        
        # Load images
        for class_name in ['real', 'fake']:
            class_dir = os.path.join(data_dir, 'train' if is_train else 'test', class_name)
            if not os.path.exists(class_dir):
                # Try alternative structure
                class_dir = os.path.join(data_dir, class_name)
            
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.image_paths)} images for {'training' if is_train else 'testing'}")
        print(f"Class distribution: Real {self.labels.count(0)}, Fake {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label

# Advanced Transformations
def get_transforms():
    train_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

# Advanced Model Architecture
class AdvancedImageClassifier(nn.Module):
    def __init__(self, model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=True):
        super(AdvancedImageClassifier, self).__init__()
        
        if model_name.startswith('efficientnet'):
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif model_name.startswith('convnext'):
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            in_features = self.backbone.head.fc.in_features
            self.backbone.head = nn.Identity()
        else:
            # Use torchvision models as fallback
            if model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        # Advanced classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# Focal Loss for imbalanced datasets
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Training function with GPU optimization and best model saving
def train_model():
    print(f"Using device: {config.DEVICE}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Model: {config.MODEL_NAME}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    try:
        train_dataset = AIImageDataset(config.DATA_PATH, transform=train_transform, is_train=True)
        val_dataset = AIImageDataset(config.DATA_PATH, transform=val_transform, is_train=False)
    except Exception as e:
        print(f"Error loading datasets with standard structure: {e}")
        print("Trying alternative directory structure...")
        
        # Alternative approach - manually split the data
        all_image_paths = []
        all_labels = []
        
        for class_name in ['real', 'fake']:
            class_dir = os.path.join(config.DATA_PATH, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(class_dir, img_name))
                        all_labels.append(0 if class_name == 'real' else 1)
        
        if len(all_image_paths) == 0:
            print("No images found! Checking directory structure...")
            # List all files and directories
            for root, dirs, files in os.walk(config.DATA_PATH):
                print(f"Directory: {root}")
                print(f"Subdirectories: {dirs[:5]}")  # First 5 subdirectories
                print(f"Files: {files[:5]}")  # First 5 files
                break
        
        print(f"Total images found: {len(all_image_paths)}")
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = CustomImageDataset(val_paths, val_labels, transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training data found! Please check your dataset path.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=min(config.NUM_WORKERS, os.cpu_count()), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=min(config.NUM_WORKERS, os.cpu_count()), pin_memory=True
    )
    
    # Initialize model
    model = AdvancedImageClassifier()
    model = model.to(config.DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()  # Uncomment for imbalanced datasets
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and config.DEVICE.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE, 
        verbose=True, 
        path=os.path.join(config.MODEL_SAVE_PATH, 'best_image_model.pth')
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    
    print("Starting training...")
    
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]')
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(config.DEVICE, non_blocking=True), labels.to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct_predictions / total_predictions
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        train_accuracy = 100. * correct_predictions / total_predictions
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Val]')
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(config.DEVICE, non_blocking=True), labels.to(config.DEVICE, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                
                current_val_acc = 100. * val_correct_predictions / val_total_predictions
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })
        
        val_accuracy = 100. * val_correct_predictions / val_total_predictions
        val_loss = val_running_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Save training history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{config.EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_accuracy': best_val_accuracy,
            }, os.path.join(config.MODEL_SAVE_PATH, 'best_image_model_full.pth'))
            print(f'  🎯 New best model saved! Validation Accuracy: {val_accuracy:.2f}%')
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_image_model_full.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Best model not found, using final model for evaluation.")
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    final_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n🎯 Final Validation Accuracy: {final_accuracy:.4f}")
    print(f"🏆 Best Validation Accuracy: {best_val_accuracy:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'final_image_model.pth'))
    
    print(f"\n✅ Training completed!")
    print(f"📁 Models saved in: {config.MODEL_SAVE_PATH}")
    print(f"🏆 Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"🎯 Final validation accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    train_model()
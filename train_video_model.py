import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import glob
import random

class VideoFeatureDataset(Dataset):
    def __init__(self, dataset_path, mode='train', max_frames=16, frame_size=224, train_ratio=0.8, val_ratio=0.1):
        """
        Dataset for video feature analysis - Fixed for FF++ structure with real/fake folders
        """
        self.dataset_path = dataset_path
        self.mode = mode
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.samples = []
        self.labels = []
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.load_dataset(train_ratio, val_ratio)
    
    def load_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """Load FF++ dataset from real/fake folders"""
        print(f"Loading {self.mode} dataset from {self.dataset_path}")
        
        real_folder = os.path.join(self.dataset_path, "real")
        fake_folder = os.path.join(self.dataset_path, "fake")
        
        # Check if folders exist
        if not os.path.exists(real_folder) or not os.path.exists(fake_folder):
            print(f"❌ Error: 'real' or 'fake' folder not found in {self.dataset_path}")
            print(f"📁 Found folders: {os.listdir(self.dataset_path)}")
            self.create_dummy_dataset()
            return
        
        # Get all video files
        real_videos = self.get_video_files(real_folder)
        fake_videos = self.get_video_files(fake_folder)
        
        print(f"Found {len(real_videos)} real videos")
        print(f"Found {len(fake_videos)} fake videos")
        
        if len(real_videos) == 0 or len(fake_videos) == 0:
            print("❌ No videos found in real/fake folders")
            self.create_dummy_dataset()
            return
        
        # Combine and shuffle
        all_samples = [(video, 0) for video in real_videos] + [(video, 1) for video in fake_videos]
        random.shuffle(all_samples)
        
        # Split dataset
        total_samples = len(all_samples)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        if self.mode == 'train':
            self.samples, self.labels = zip(*all_samples[:train_end])
        elif self.mode == 'val':
            self.samples, self.labels = zip(*all_samples[train_end:val_end])
        else:  # test
            self.samples, self.labels = zip(*all_samples[val_end:])
        
        print(f"✅ Loaded {len(self.samples)} {self.mode} samples")
    
    def get_video_files(self, folder_path):
        """Get all video files from folder"""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
        video_files = []
        
        for extension in video_extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, "**", extension), recursive=True))
            video_files.extend(glob.glob(os.path.join(folder_path, extension)))
        
        return video_files
    
    def extract_frames(self, video_path):
        """Extract frames from video"""
        frames = []
        
        try:
            # For dummy videos, return random frames
            if "dummy" in video_path:
                for _ in range(self.max_frames):
                    frame = np.random.randint(0, 255, (self.frame_size, self.frame_size, 3), dtype=np.uint8)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                return frames
            
            # Real video processing
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                return self.create_random_frames()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"❌ Video has 0 frames: {video_path}")
                return self.create_random_frames()
            
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, min(self.max_frames, total_frames), dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                else:
                    # Add random frame if reading fails
                    frame = np.random.randint(0, 255, (self.frame_size, self.frame_size, 3), dtype=np.uint8)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Error extracting frames from {video_path}: {e}")
            frames = self.create_random_frames()
        
        return frames
    
    def create_random_frames(self):
        """Create random frames as fallback"""
        frames = []
        for _ in range(self.max_frames):
            frame = np.random.randint(0, 255, (self.frame_size, self.frame_size, 3), dtype=np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
        return frames
    
    def create_dummy_dataset(self):
        """Create dummy dataset for testing when no videos are available"""
        print("⚠️ Creating dummy dataset for testing...")
        for i in range(100):
            self.samples.append(f"dummy_video_{i}.mp4")
            self.labels.append(i % 2)  # Alternate between real and fake
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Apply transformations
        transformed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            transformed_frames.append(frame)
        
        # Stack frames into tensor
        if len(transformed_frames) > 0:
            video_tensor = torch.stack(transformed_frames)
        else:
            # Create empty tensor if no frames
            video_tensor = torch.zeros(self.max_frames, 3, self.frame_size, self.frame_size)
        
        return video_tensor, label, video_path

class VideoFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_classes=2):
        super(VideoFeatureExtractor, self).__init__()
        
        # Frame feature extractor
        self.frame_extractor = timm.create_model(
            backbone_name, 
            pretrained=True, 
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        self.feature_dim = self.frame_extractor.num_features
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, frames, channels, height, width)
        batch_size, num_frames, C, H, W = x.shape
        
        # Extract features for each frame
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            features = self.frame_extractor(frame)
            frame_features.append(features)
        
        # Stack frame features: (batch_size, num_frames, feature_dim)
        temporal_features = torch.stack(frame_features, dim=1)
        
        # Temporal pooling
        pooled_features = self.temporal_pool(temporal_features.transpose(1, 2))
        pooled_features = pooled_features.squeeze(-1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

class VideoForensicsTrainer:
    def __init__(self, dataset_path, model_save_path):
        # GPU optimization
        self.device = self.setup_device()
        self.scaler = torch.cuda.amp.GradScaler()  # Fixed GradScaler
        
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        
        # Create model save directory
        os.makedirs(model_save_path, exist_ok=True)
        
        # Initialize model
        self.model = VideoFeatureExtractor(num_classes=2).to(self.device)
        
        # Use DataParallel for multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'best_val_acc': 0.0, 'best_epoch': 0
        }
    
    def setup_device(self):
        """Setup and verify GPU usage"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Print GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            print(f"✅ CUDA version: {torch.version.cuda}")
            
            # Enable benchmark mode for faster convolutions
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
            print("❌ No GPU detected, using CPU (training will be slow)")
        
        return device
    
    def create_data_loaders(self, batch_size=4, num_workers=2):
        """Create data loaders for training, validation, and testing"""
        # Use smaller batch size for video data
        train_dataset = VideoFeatureDataset(self.dataset_path, mode='train')
        val_dataset = VideoFeatureDataset(self.dataset_path, mode='val')
        test_dataset = VideoFeatureDataset(self.dataset_path, mode='test')
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Val samples: {len(val_dataset)}")
        print(f"📊 Test samples: {len(test_dataset)}")
    
    def train_epoch(self, epoch):
        """Train for one epoch with mixed precision - FIXED AUTOCAST"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_idx, (videos, labels, _) in enumerate(pbar):
            videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # FIXED: Use new autocast API
            with torch.amp.autocast(device_type='cuda'):  # Fixed autocast
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch - FIXED AUTOCAST"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} Validation')
            for batch_idx, (videos, labels, _) in enumerate(pbar):
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                # FIXED: Use new autocast API
                with torch.amp.autocast(device_type='cuda'):  # Fixed autocast
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate additional metrics
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)
        
        print(f'📊 Validation - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, '
              f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        # Get model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'val_acc': self.history['val_acc'][-1]
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_save_path, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.model_save_path, 'best_video_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved with validation accuracy: {self.history['val_acc'][-1]:.2f}%")
    
    def train(self, epochs=20, patience=5):
        """Main training loop"""
        print("🚀 Starting video forensics training...")
        print(f"🎯 Using device: {self.device}")
        
        self.create_data_loaders(batch_size=4, num_workers=2)  # Smaller batch size for videos
        
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            print(f'📈 Training - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'📉 Learning rate: {current_lr:.2e}')
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                self.history['best_val_acc'] = best_val_acc
                self.history['best_epoch'] = epoch
                epochs_no_improve = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                epochs_no_improve += 1
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save training history
        history_path = os.path.join(self.model_save_path, 'training_history.pkl')
        joblib.dump(self.history, history_path)
        
        # Load best model for final evaluation
        self.load_best_model()
        
        return self.history
    
    def load_best_model(self):
        """Load the best saved model"""
        best_model_path = os.path.join(self.model_save_path, 'best_video_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            # Handle DataParallel wrapping
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            print("✅ Loaded best model for evaluation")
    
    def evaluate(self):
        """Evaluate on test set - FIXED AUTOCAST"""
        if not hasattr(self, 'test_loader'):
            self.create_data_loaders()
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for videos, labels, _ in pbar:
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                # FIXED: Use new autocast API
                with torch.amp.autocast(device_type='cuda'):  # Fixed autocast
                    outputs = self.model(videos)
                    probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        print(f"\n{'='*50}")
        print("🎯 FINAL TEST RESULTS")
        print(f"{'='*50}")
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ F1 Score: {f1:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall: {recall:.4f}")
        
        # Save test results
        test_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        results_path = os.path.join(self.model_save_path, 'test_results.pkl')
        joblib.dump(test_results, results_path)
        
        return test_results

def check_dataset_structure(dataset_path):
    """Check and display dataset structure"""
    print("🔍 Checking dataset structure...")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    real_folder = os.path.join(dataset_path, "real")
    fake_folder = os.path.join(dataset_path, "fake")
    
    print(f"📁 Dataset path: {dataset_path}")
    print(f"📁 Contents: {os.listdir(dataset_path)}")
    
    if os.path.exists(real_folder):
        real_videos = glob.glob(os.path.join(real_folder, "**/*.mp4"), recursive=True)
        real_videos += glob.glob(os.path.join(real_folder, "*.mp4"))
        print(f"📹 Real videos found: {len(real_videos)}")
        
        if real_videos:
            print(f"   Sample real videos: {real_videos[:3]}")
    else:
        print("❌ Real folder not found")
    
    if os.path.exists(fake_folder):
        fake_videos = glob.glob(os.path.join(fake_folder, "**/*.mp4"), recursive=True)
        fake_videos += glob.glob(os.path.join(fake_folder, "*.mp4"))
        print(f"📹 Fake videos found: {len(fake_videos)}")
        
        if fake_videos:
            print(f"   Sample fake videos: {fake_videos[:3]}")
    else:
        print("❌ Fake folder not found")
    
    return True

def train_video_model():
    """Main function to train video forensics model"""
    dataset_path = r"C:\Users\padma\Downloads\df\archive (3)\FF++"
    model_save_path = r"C:\Users\padma\OneDrive\Desktop\Models\ai-forensic-webapp\models"
    
    # Check dataset structure first
    if not check_dataset_structure(dataset_path):
        print("❌ Dataset structure issue detected!")
        return None, None, None
    
    # Initialize trainer
    trainer = VideoForensicsTrainer(dataset_path, model_save_path)
    
    # Train model (fewer epochs for testing)
    history = trainer.train(epochs=20, patience=5)
    
    # Evaluate on test set
    test_results = trainer.evaluate()
    
    print("\n🎉 Video forensics model training completed!")
    return trainer, history, test_results

if __name__ == "__main__":
    print("🎬 Video Forensics Model Training")
    print("=" * 50)
    
    # Train the model
    trainer, history, test_results = train_video_model()
    
    if trainer is not None:
        print(f"\n💾 Best model saved to: {trainer.model_save_path}")
        print(f"📊 Best validation accuracy: {history['best_val_acc']:.2f}%")
        print(f"📈 Training completed successfully!")
    else:
        print("❌ Training failed!")
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK downloads may require manual setup")

# Configuration
class Config:
    DATA_PATH = r"C:\Users\padma\Downloads\archive (6)"
    MODEL_SAVE_PATH = "models"
    BATCH_SIZE = 64  # Increased for GPU
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MAX_FEATURES = 5000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True  # Mixed precision training

config = Config()

class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer=None, max_features=5000):
        self.texts = texts
        self.labels = labels
        
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            self.features = self.vectorizer.fit_transform(texts).toarray()
        else:
            self.vectorizer = vectorizer
            self.features = vectorizer.transform(texts).toarray()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

class AdvancedTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512, 256], num_classes=2, dropout_rate=0.3):
        super(AdvancedTextClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Binary classification head (AI vs Human)
        self.binary_classifier = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate//2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.binary_classifier(features)
        return output

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits, but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.!?]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

def setup_gpu():
    """Setup GPU optimization settings"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎯 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
    else:
        device = torch.device('cpu')
        print("❌ Using CPU - GPU not available")
    
    return device

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    
    # Check what files are available
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return {}
    
    files = os.listdir(data_path)
    print(f"Available files: {files}")
    
    datasets = {}
    
    # Look for CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    
    for csv_file in csv_files:
        try:
            file_path = os.path.join(data_path, csv_file)
            print(f"Loading {file_path}...")
            df = pd.read_csv(file_path)
            datasets[csv_file] = df
            print(f"Loaded {csv_file}: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"Sample data:\n{df.head()}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return datasets

def create_training_data(datasets):
    """Create unified training data from multiple datasets"""
    all_texts = []
    all_binary_labels = []  # 0: Human, 1: AI
    
    preprocessor = TextPreprocessor()
    
    for filename, df in datasets.items():
        print(f"Processing {filename}...")
        
        # Determine the structure of each dataset
        text_column = None
        label_column = None
        
        # Common column name patterns
        text_candidates = ['text', 'content', 'message', 'tweet', 'post', 'article', 'response', 'prompt']
        label_candidates = ['label', 'source', 'generator', 'type', 'class', 'is_ai', 'platform', 'category']
        
        for col in df.columns:
            if col.lower() in text_candidates:
                text_column = col
            if col.lower() in label_candidates:
                label_column = col
        
        if text_column is None:
            # Try to find any string column
            for col in df.columns:
                if df[col].dtype == 'object' and col != label_column:
                    text_column = col
                    break
        
        if text_column is None:
            print(f"No suitable text column found in {filename}")
            continue
        
        print(f"Using text column: {text_column}")
        print(f"Using label column: {label_column}")
        
        # Process each row
        for idx, row in df.iterrows():
            text = row[text_column]
            
            if pd.isna(text):
                continue
            
            # Clean text
            cleaned_text = preprocessor.clean_text(str(text))
            
            if len(cleaned_text.split()) < 3:  # Skip very short texts
                continue
            
            all_texts.append(cleaned_text)
            
            # Determine labels
            if label_column and label_column in row and not pd.isna(row[label_column]):
                label_value = str(row[label_column]).lower()
                
                # Binary classification (AI vs Human)
                if any(ai_indicator in label_value for ai_indicator in ['ai', 'gpt', 'chatgpt', 'llm', 'generated', 'synthetic', 'machine', 'fake', '1', 'true']):
                    all_binary_labels.append(1)  # AI
                elif any(human_indicator in label_value for human_indicator in ['human', 'real', 'organic', 'natural', '0', 'false']):
                    all_binary_labels.append(0)  # Human
                else:
                    # Default based on filename or random
                    if 'ai' in filename.lower() or 'fake' in filename.lower():
                        all_binary_labels.append(1)
                    else:
                        all_binary_labels.append(0)
            
            else:
                # If no label column, use filename as hint
                if 'ai' in filename.lower() or 'fake' in filename.lower() or 'gpt' in filename.lower():
                    all_binary_labels.append(1)
                else:
                    all_binary_labels.append(0)
    
    return all_texts, all_binary_labels

def train_model():
    """Main training function with GPU support"""
    
    # Setup GPU
    device = setup_gpu()
    
    # Create model directory
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Load data
    datasets = load_and_preprocess_data(config.DATA_PATH)
    
    if not datasets:
        print("No datasets found! Creating synthetic data for demonstration...")
        # Create more diverse synthetic data
        synthetic_texts = [
            # Human texts (more natural, varied)
            "I was thinking about going to the park today, but the weather looks a bit cloudy.",
            "Honestly, I'm not sure what to have for dinner tonight. Maybe pasta?",
            "The meeting went longer than expected, but we managed to cover all the important points.",
            "I love how the sunlight filters through the trees in the morning.",
            "Can you believe it's already Friday? This week flew by so quickly!",
            "My dog keeps barking at the mailman every single day without fail.",
            "I need to remember to buy groceries on my way home from work today.",
            "The book I'm reading is getting really interesting in the middle chapters.",
            
            # AI texts (more structured, formal)
            "Based on the information provided, I can assist you with that query.",
            "As an AI language model, I'm designed to provide helpful responses.",
            "The user has requested information about the specified topic.",
            "I can generate a comprehensive response to address your question.",
            "According to the parameters set, the output should meet your requirements.",
            "The system is processing your request and will provide a suitable response.",
            "Based on the available data, I can offer the following information.",
            "This response is generated to fulfill the user's specific inquiry."
        ]
        synthetic_binary = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # 0: Human, 1: AI
        
        all_texts = synthetic_texts
        all_binary_labels = synthetic_binary
    else:
        # Process real data
        all_texts, all_binary_labels = create_training_data(datasets)
    
    print(f"Total samples: {len(all_texts)}")
    print(f"Binary labels distribution: {Counter(all_binary_labels)}")
    
    if len(all_texts) == 0:
        print("❌ No training data available!")
        return
    
    # Check if we have both classes
    unique_labels = set(all_binary_labels)
    if len(unique_labels) < 2:
        print(f"❌ Only one class found: {unique_labels}. Need both classes for training.")
        print("Adding synthetic data to balance classes...")
        # Add synthetic examples of the missing class
        preprocessor = TextPreprocessor()
        if 0 not in unique_labels:  # Missing human examples
            human_texts = [
                "I really enjoyed walking in the park yesterday evening.",
                "What do you think about trying that new restaurant?",
                "The traffic was terrible this morning on my way to work.",
                "I can't decide what movie to watch tonight.",
                "My coffee this morning was exactly what I needed."
            ]
            all_texts.extend(human_texts)
            all_binary_labels.extend([0, 0, 0, 0, 0])
        if 1 not in unique_labels:  # Missing AI examples
            ai_texts = [
                "I can provide information based on the available data.",
                "The system is designed to process user inputs efficiently.",
                "Based on the analysis, the results indicate positive outcomes.",
                "This response is generated to address your specific query.",
                "The model can assist with various types of requests."
            ]
            all_texts.extend(ai_texts)
            all_binary_labels.extend([1, 1, 1, 1, 1])
    
    print(f"Final dataset size: {len(all_texts)}")
    print(f"Final label distribution: {Counter(all_binary_labels)}")
    
    # Split data
    X_train, X_test, y_binary_train, y_binary_test = train_test_split(
        all_texts, all_binary_labels, 
        test_size=0.2, random_state=42, stratify=all_binary_labels
    )
    
    X_train, X_val, y_binary_train, y_binary_val = train_test_split(
        X_train, y_binary_train, 
        test_size=0.2, random_state=42, stratify=y_binary_train
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_binary_train, max_features=config.MAX_FEATURES)
    val_dataset = TextDataset(X_val, y_binary_val, vectorizer=train_dataset.vectorizer)
    test_dataset = TextDataset(X_test, y_binary_test, vectorizer=train_dataset.vectorizer)
    
    # Data loaders with GPU optimizations
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # Initialize model and move to GPU
    input_size = train_dataset.features.shape[1]
    model = AdvancedTextClassifier(input_size=input_size)
    model = model.to(device)
    
    # Loss functions and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and device.type == 'cuda' else None
    
    # Training history
    train_losses = []
    val_accuracies = []
    
    print("Starting training with GPU...")
    
    best_val_accuracy = 0.0
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            # Move data to GPU
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).squeeze()
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True).squeeze()
                
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true, val_predictions)
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{config.EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'vectorizer': train_dataset.vectorizer,
                'input_size': input_size
            }, os.path.join(config.MODEL_SAVE_PATH, 'best_text_model.pth'))
            print(f"🎯 New best model saved with validation accuracy: {val_accuracy:.4f}")
    
    # Load best model for testing
    checkpoint = torch.load(os.path.join(config.MODEL_SAVE_PATH, 'best_text_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test the model
    model.eval()
    test_predictions = []
    test_true = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).squeeze()
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true.extend(batch_labels.cpu().numpy())
    
    test_accuracy = accuracy_score(test_true, test_predictions)
    print(f"\n🎯 Final Test Accuracy: {test_accuracy:.4f}")
    
    # Handle case where only one class is predicted
    unique_predicted = set(test_predictions)
    unique_true = set(test_true)
    
    print(f"Unique classes in predictions: {unique_predicted}")
    print(f"Unique classes in true labels: {unique_true}")
    
    print("\nClassification Report:")
    # Use labels parameter to ensure both classes are included
    labels = sorted(list(unique_true.union(unique_predicted)))
    target_names = ['Human', 'AI']
    
    # Only include target names for existing labels
    report_target_names = [target_names[i] for i in labels if i < len(target_names)]
    
    print(classification_report(test_true, test_predictions, 
                               labels=labels, 
                               target_names=report_target_names,
                               zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(test_true, test_predictions, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=report_target_names, 
                yticklabels=report_target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'text_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save vectorizer and metadata
    joblib.dump(train_dataset.vectorizer, os.path.join(config.MODEL_SAVE_PATH, 'text_vectorizer.pkl'))
    
    metadata = {
        'input_size': input_size,
        'test_accuracy': test_accuracy,
        'feature_names': train_dataset.vectorizer.get_feature_names_out().tolist() if hasattr(train_dataset.vectorizer, 'get_feature_names_out') else []
    }
    
    joblib.dump(metadata, os.path.join(config.MODEL_SAVE_PATH, 'model_metadata.pkl'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'text_training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Model training completed!")
    print(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"🎯 Test accuracy: {test_accuracy:.4f}")
    print(f"📁 Models saved in: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    train_model()
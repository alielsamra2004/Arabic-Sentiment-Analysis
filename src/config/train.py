# Configuration constants - Edit these values to change training parameters
DEFAULT_CONFIG = {
    "model_name": "UBC-NLP/MARBERT",
    "max_train_samples": 1000,  # Total number of training samples
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "batch_size": 32,
    "max_length": 128,
    "output_dir": "outputs/ar_reviews"
}

"""
Training script for Arabic sentiment analysis using MARBERT.

This script handles training a MARBERT model on the Arabic reviews dataset
for sentiment classification.

Default configuration:
- Total training samples: 600 (200 per class)
- Model: UBC-NLP/MARBERT
- Learning rate: 2e-5
- Epochs: 5
- Batch size: 32
- Max sequence length: 128
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import json
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('.')  # Add project root to Python path
from src.config.data_loader1 import load_arabic_reviews

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def subsample_dataset(dataset, max_samples=None):
    """
    Create a subset of the dataset with balanced labels.
    
    Args:
        dataset: Original dataset
        max_samples: Maximum number of samples to use (per class)
        
    Returns:
        Subsampled dataset with balanced classes
    """
    if max_samples is None:
        return dataset
    
    # Get indices for each label
    label_indices = {}
    for idx, label in enumerate(dataset["label"]):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
    
    # Calculate samples per class
    samples_per_class = max_samples // len(label_indices)
    print(f"\nSampling {samples_per_class} examples per class")
    
    # Sample indices for each class
    selected_indices = []
    for label, indices in label_indices.items():
        if len(indices) > samples_per_class:
            # Randomly sample from this class
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            # Take all samples if we have fewer than needed
            selected = indices
        selected_indices.extend(selected)
    
    # Shuffle the selected indices
    np.random.shuffle(selected_indices)
    
    # Create new dataset with selected indices
    return dataset.select(selected_indices)

class SentimentClassifier:
    def __init__(self, model_name="UBC-NLP/MARBERT", num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.model.to(device)
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, texts, labels=None, max_length=128):
        """Prepare data for model input."""
        # Add special tokens
        texts = ["[CLS] " + str(text) + " [SEP]" for text in texts]
        
        # Tokenize texts
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        
        # Convert tokens to ids and pad
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", 
                                value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        
        # Create attention masks
        attention_masks = [[float(i != self.tokenizer.pad_token_id) for i in ids] for ids in input_ids]
        
        # Convert to tensors
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        
        if labels is not None:
            labels = torch.tensor(labels)
            return inputs, masks, labels
        return inputs, masks
    
    def fit_label_encoder(self, labels):
        """Fit label encoder on training labels."""
        self.label_encoder.fit(labels)
        print(f"Found labels: {self.label_encoder.classes_}")
        
    def encode_labels(self, labels):
        """Encode string labels to numeric indices."""
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, label_ids):
        """Decode numeric indices back to string labels."""
        return self.label_encoder.inverse_transform(label_ids)
    
    def save(self, output_dir):
        """Save model, tokenizer, and label encoder."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # Save label encoder classes
        np.save(os.path.join(output_dir, "label_encoder_classes.npy"), 
                self.label_encoder.classes_)
        print(f"Model and tokenizer saved to {output_dir}")
        print(f"Label classes saved: {self.label_encoder.classes_}")

def train_epoch(model, train_dataloader, optimizer, scheduler):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_dataloader)

def evaluate(model, eval_dataloader, label_encoder):
    """Evaluate model performance."""
    model.eval()
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(eval_dataloader, desc="Evaluating")
    for batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # Convert numeric predictions back to original labels for interpretable metrics
    pred_labels = label_encoder.inverse_transform(all_preds)
    true_labels = label_encoder.inverse_transform(all_labels)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    precision = precision_score(true_labels, pred_labels, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }

def main():
    parser = argparse.ArgumentParser(description="Train MARBERT for Arabic sentiment analysis")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"],
                      help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                      help="Output directory for saving models")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                      help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                      help="Training batch size")
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"],
                      help="Maximum sequence length")
    parser.add_argument("--max_train_samples", type=int, default=DEFAULT_CONFIG["max_train_samples"],
                      help="Maximum number of training samples (total, divided equally among classes)")
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    datasets = load_arabic_reviews()
    if datasets is None:
        print("Failed to load dataset!")
        return
    
    # Subsample training data
    print(f"\nSubsampling training data to {args.max_train_samples} total samples...")
    datasets["train"] = subsample_dataset(datasets["train"], args.max_train_samples)
    
    # Initialize classifier
    print("\nInitializing classifier...")
    unique_labels = set(datasets["train"]["label"])
    print(f"Found {len(unique_labels)} unique labels in training data: {unique_labels}")
    classifier = SentimentClassifier(args.model_name, num_labels=len(unique_labels))
    
    # Prepare labels
    print("\nPreparing labels...")
    classifier.fit_label_encoder(datasets["train"]["label"])
    train_labels = classifier.encode_labels(datasets["train"]["label"])
    val_labels = classifier.encode_labels(datasets["validation"]["label"])
    
    # Prepare data
    print("\nPreparing data...")
    train_inputs, train_masks, train_labels = classifier.prepare_data(
        datasets["train"]["input_text"], train_labels, args.max_length)
    val_inputs, val_masks, val_labels = classifier.prepare_data(
        datasets["validation"]["input_text"], val_labels, args.max_length)
    
    print(f"\nTraining samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    
    # Print class distribution
    print("\nClass distribution in training data:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label_id, count in zip(unique, counts):
        label_name = classifier.decode_labels([label_id])[0]
        print(f"{label_name}: {count} samples")
    
    # Create dataloaders
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)
    
    # Prepare optimizer and scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    optimizer = AdamW(classifier.model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps * 0.1,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(classifier.model, train_dataloader, optimizer, scheduler)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(classifier.model, val_dataloader, classifier.label_encoder)
        print(f"\nValidation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            print(f"\nNew best F1 score: {best_f1:.4f}")
            model_path = os.path.join(args.output_dir, "best_model")
            classifier.save(model_path)
            
            # Save metrics
            with open(os.path.join(model_path, "metrics.json"), "w") as f:
                json.dump(metrics, f)
    
    print("\nTraining completed!")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_labels = classifier.encode_labels(datasets["test"]["label"])
    test_inputs, test_masks, test_labels = classifier.prepare_data(
        datasets["test"]["input_text"], test_labels, args.max_length)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    
    test_metrics = evaluate(classifier.model, test_dataloader, classifier.label_encoder)
    print("\nTest set metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save test metrics
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f)

if __name__ == "__main__":
    main() 
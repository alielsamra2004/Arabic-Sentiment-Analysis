"""
Generate confusion matrix from the best saved model.
This script loads the saved model and creates confusion matrix visualizations
without needing to retrain the model.
"""

import os
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('.')  # Add project root to Python path
from src.config.data_loader1 import load_arabic_reviews

def load_saved_model(model_dir):
    """Load the saved model and tokenizer."""
    print(f"\nLoading model from {model_dir}")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    # Load label encoder classes
    label_classes = np.load(os.path.join(model_dir, "label_encoder_classes.npy"))
    print(f"Found labels: {label_classes}")
    
    return model, tokenizer, label_classes

def prepare_data(texts, tokenizer, max_length=128):
    """Prepare texts for model input."""
    # Add special tokens
    texts = ["[CLS] " + str(text) + " [SEP]" for text in texts]
    
    # Tokenize
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    
    # Convert to ids and pad
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    from keras.preprocessing.sequence import pad_sequences
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long",
                            value=tokenizer.pad_token_id, truncating="post", padding="post")
    
    # Create attention masks
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ids] for ids in input_ids]
    
    return torch.tensor(input_ids), torch.tensor(attention_masks)

def get_predictions(model, dataloader, device):
    """Get model predictions."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask = batch
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    return np.array(all_preds)

def plot_confusion_matrix(true_labels, pred_labels, label_names, output_dir):
    """Create and save confusion matrix visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Create display
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names
    )
    
    # Plot with custom styling
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix for Arabic Sentiment Analysis')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add value annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    ha='center', va='center')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw numbers
    cm_dict = {
        'matrix': cm.tolist(),
        'labels': label_names.tolist(),
        'row_totals': cm.sum(axis=1).tolist(),
        'column_totals': cm.sum(axis=0).tolist(),
        'accuracy_per_class': (cm.diagonal() / cm.sum(axis=1)).tolist()
    }
    
    with open(os.path.join(output_dir, "confusion_matrix.json"), 'w') as f:
        json.dump(cm_dict, f, indent=2)
    
    # Print analysis
    print("\nConfusion Matrix Analysis:")
    print("-" * 50)
    for i, label in enumerate(label_names):
        accuracy = cm[i, i] / cm[i].sum() * 100
        print(f"{label}:")
        print(f"  Total samples: {cm[i].sum()}")
        print(f"  Correct predictions: {cm[i, i]}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print()

def main():
    # Configuration
    model_dir = "outputs/ar_reviews/best_model"
    output_dir = "outputs/ar_reviews/confusion_matrix"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and data
    model, tokenizer, label_classes = load_saved_model(model_dir)
    model.to(device)
    
    print("\nLoading dataset...")
    datasets = load_arabic_reviews()
    if datasets is None:
        print("Failed to load dataset!")
        return
    
    # Prepare test data
    print("\nPreparing test data...")
    test_inputs, test_masks = prepare_data(datasets["test"]["input_text"], tokenizer)
    test_data = TensorDataset(test_inputs, test_masks)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # Get predictions
    print("\nGetting predictions...")
    predictions = get_predictions(model, test_dataloader, device)
    
    # Create confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        datasets["test"]["label"],
        [label_classes[pred] for pred in predictions],
        label_classes,
        output_dir
    )
    
    print(f"\nConfusion matrix saved to {output_dir}")

if __name__ == "__main__":
    main() 
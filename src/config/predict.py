"""
CLI tool for English text sentiment analysis using translation and MARBERT model.

This script:
1. Translates English input to Arabic
2. Uses a fine-tuned MARBERT model to predict sentiment
3. Visualizes and logs the results
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import torch.nn.functional as F
import arabic_reshaper
from bidi.algorithm import get_display

# Constants
MODEL_PATH = "outputs/ar_reviews/best_model"
LOG_FILE = "outputs/prediction_log.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging():
    """Initialize logging directory and CSV file if they don't exist."""
    os.makedirs("outputs", exist_ok=True)
    
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp",
            "english_text",
            "arabic_translation",
            "predicted_sentiment",
            "probabilities"
        ]).to_csv(LOG_FILE, index=False)

def load_model():
    """Load the pre-trained MARBERT model and tokenizer."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        model.to(DEVICE)
        model.eval()
        
        # Load class labels
        labels = np.load(os.path.join(MODEL_PATH, "label_encoder_classes.npy"))
        
        return model, tokenizer, labels
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def translate_to_arabic(text):
    """Translate English text to Arabic using Helsinki-NLP model."""
    try:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
        translation = translator(text, max_length=128)[0]['translation_text']
        return translation
    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}")

def predict_sentiment(model, tokenizer, arabic_text, labels):
    """Predict sentiment of Arabic text using MARBERT model."""
    try:
        # Tokenize input
        inputs = tokenizer(
            arabic_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1)
        
        # Get predicted class and probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probs = probabilities[0].cpu().numpy()
        
        return {
            "sentiment": labels[predicted_class],
            "probabilities": {label: float(prob) for label, prob in zip(labels, probs)}
        }
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

def log_prediction(english_text, arabic_text, results):
    """Log prediction results to CSV file."""
    timestamp = datetime.now().isoformat()
    probs_str = "; ".join([
        f"{label}: {prob:.4f}"
        for label, prob in results["probabilities"].items()
    ])
    
    new_row = pd.DataFrame([{
        "timestamp": timestamp,
        "english_text": english_text,
        "arabic_translation": arabic_text,
        "predicted_sentiment": results["sentiment"],
        "probabilities": probs_str
    }])
    
    new_row.to_csv(LOG_FILE, mode='a', header=False, index=False)

def visualize_probabilities(probabilities, arabic_text):
    """Create a bar chart of sentiment probabilities with Arabic text."""
    # Set up RTL text rendering
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 8))  # Made figure taller to accommodate text
    
    # Create subplot for the bar chart
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    bars = plt.bar(labels, values)
    
    # Customize the plot
    plt.title("Sentiment Prediction Probabilities")
    plt.xlabel("Sentiment")
    plt.ylabel("Probability")
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    # Add Arabic text in a separate subplot
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.axis('off')  # Hide axes
    
    # Create a text box with RTL direction
    # Reshape and reorder Arabic text for proper display
    reshaped_text = arabic_reshaper.reshape(arabic_text)
    bidi_text = get_display(reshaped_text)
    
    plt.text(
        0.5, 0.5,
        bidi_text,
        fontsize=14,
        ha='center',
        va='center',
        fontfamily='Arial',  # Use a font that supports Arabic
        rotation=0,
        wrap=True
    )
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze sentiment of English text")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="English text to analyze"
    )
    args = parser.parse_args()
    
    # Validate input
    if not args.text.strip():
        print("Error: Please provide non-empty text input")
        return
    
    try:
        # Initialize logging
        setup_logging()
        
        # Load model and components
        print("Loading model...")
        model, tokenizer, labels = load_model()
        
        # Translate text
        print("Translating to Arabic...")
        arabic_text = translate_to_arabic(args.text)
        print(f"Arabic translation: {arabic_text}")
        
        # Predict sentiment
        print("\nAnalyzing sentiment...")
        results = predict_sentiment(model, tokenizer, arabic_text, labels)
        
        # Log results
        log_prediction(args.text, arabic_text, results)
        
        # Display results
        print(f"\nPredicted sentiment: {results['sentiment']}")
        print("\nProbabilities:")
        for label, prob in results["probabilities"].items():
            print(f"{label}: {prob:.4f}")
        
        # Visualize results with Arabic text
        visualize_probabilities(results["probabilities"], arabic_text)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main() 
"""
Arabic Sentiment Classifier Interface

This module provides a Gradio web interface for Arabic sentiment classification
using the fine-tuned MARBERT model.
"""

import os
import csv
import torch
import gradio as gr
import numpy as np
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch.nn.functional as F
from typing import Dict, Any

# Constants
MODEL_PATH = "outputs/ar_reviews/best_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "outputs/predictions_log.csv"
os.makedirs("outputs", exist_ok=True)

# Write header once
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "input_text", "predicted_sentiment", "probabilities", "translation"])

def log_prediction(text, sentiment, probabilities, translation):
    timestamp = datetime.now().isoformat()
    probs_str = "; ".join([f"{label}: {round(prob, 4)}" for label, prob in probabilities.items()])
    with open(LOG_FILE, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, text, sentiment, probs_str, translation])

class ArabicSentimentClassifier:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(DEVICE)
            self.model.eval()

            self.labels = np.load(os.path.join(model_path, "label_encoder_classes.npy"))
            print(f"Loaded labels: {self.labels}")

            try:
                self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")
            except Exception as e:
                print(f"Warning: Could not load translation model: {e}")
                self.translator = None

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def translate_to_english(self, text: str) -> str:
        try:
            if self.translator is not None:
                translation = self.translator(text, max_length=128)[0]['translation_text']
                return translation
            return "Translation not available"
        except Exception as e:
            return f"Translation error: {str(e)}"

    def classify_text(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Please enter some text to analyze.")

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            probs = probabilities[0].cpu().numpy()
            translation = self.translate_to_english(text)

            results = {
                "sentiment": self.labels[predicted_class],
                "probabilities": {
                    label: float(prob) for label, prob in zip(self.labels, probs)
                },
                "translation": translation,
                "input_text": text
            }

            return results
        except Exception as e:
            raise RuntimeError(f"Error processing text: {str(e)}")

def create_interface():
    try:
        classifier = ArabicSentimentClassifier(MODEL_PATH)
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        return None

    def process_text(text: str) -> tuple:
        if not text or not text.strip():
            return (
                "⚠️ Please enter some text to analyze.",
                None,
                None,
                None
            )

        try:
            results = classifier.classify_text(text)
            probs = results["probabilities"]

            # ✅ Log to CSV
            log_prediction(
                text=text,
                sentiment=results["sentiment"],
                probabilities=probs,
                translation=results["translation"]
            )

            sentiment_colored = f"<span style='color:{'green' if results['sentiment']=='positive' else 'red' if results['sentiment']=='negative' else 'orange'}; font-weight:bold;'>{results['sentiment'].capitalize()}</span>"

            return (
                gr.update(visible=False),
                sentiment_colored,
                probs,
                results["translation"]
            )
        except Exception as e:
            return (
                f"❌ Error: {str(e)}",
                None,
                None,
                None
            )

    with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {max-width: 1000px; margin: auto;}
    .example-text {font-size: 0.9em; color: #666;}
    .sentiment-label {font-size: 1.2em; font-weight: bold;}
""") as iface:
        gr.Markdown(
            """
            # Arabic Sentiment Classifier 🇪🇬

            This app analyzes the sentiment of Arabic text using a fine-tuned MARBERT model.
            Enter your text in Arabic, and the model will classify it as positive, negative, or mixed,
            along with confidence scores and an English translation.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Enter an Arabic review:",
                    placeholder="اكتب النص هنا...",
                    lines=3
                )
                error_message = gr.Markdown(visible=False)

            with gr.Column(scale=3):
                sentiment_label = gr.Markdown(label="Predicted Sentiment")
                prob_plot = gr.Label(label="Confidence Scores")
                translation_text = gr.Markdown(label="English Translation")

        with gr.Row():
            submit_btn = gr.Button("Analyze", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Examples(
            examples=[
                ["هذا المنتج رائع جداً وأنا سعيد بشرائه"],
                ["للأسف المنتج سيء ولا أنصح به"],
                ["المنتج جيد ولكن السعر مرتفع قليلاً"],
                ["الخدمة ممتازة والموظفين محترمين جداً"],
                ["تجربة سيئة جداً، لن أعود مرة أخرى"]
            ],
            inputs=input_text,
            label="Example Reviews"
        )

        submit_btn.click(
            fn=process_text,
            inputs=input_text,
            outputs=[error_message, sentiment_label, prob_plot, translation_text]
        )

        clear_btn.click(
            fn=lambda: (
                "",
                gr.update(visible=False),
                None,
                None,
                None
            ),
            inputs=[],
            outputs=[input_text, error_message, sentiment_label, prob_plot, translation_text]
        )

    return iface

def main():
    iface = create_interface()
    if iface:
        iface.launch(share=True)
    else:
        print("Failed to create interface")

if __name__ == "__main__":
    main()

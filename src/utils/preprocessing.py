"""
Arabic text preprocessing utilities for sentiment analysis.

This module provides functions for cleaning and normalizing Arabic text data,
including removal of diacritics, elongation, and special characters.
"""

import re
import os
import pandas as pd
from typing import Dict, Union
from datasets import DatasetDict

def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel) from text."""
    if not isinstance(text, str):
        return ""
    # Arabic diacritics range
    return re.sub(r'[\u064B-\u065F\u0670]', '', text)

def remove_elongation(text: str) -> str:
    """Reduce repeated characters to maximum of 2 repetitions."""
    if not isinstance(text, str):
        return ""
    # Replace 3 or more occurrences with 2
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def clean_text(text: str) -> str:
    """
    Clean Arabic text by:
    - Removing URLs
    - Removing mentions and hashtags
    - Removing non-Arabic letters except numbers and whitespace
    - Normalizing whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Keep Arabic letters, numbers, and whitespace
    text = re.sub(r'[^\u0600-\u06FF\s0-9]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def preprocess_arabic_text(text: str) -> str:
    """Apply all preprocessing steps to Arabic text."""
    if not isinstance(text, str):
        return ""
    
    text = clean_text(text)
    text = remove_diacritics(text)
    text = remove_elongation(text)
    
    return text

def preprocess_and_save(dataset: DatasetDict, dataset_name: str) -> DatasetDict:
    """
    Preprocess dataset and save to disk.
    
    Args:
        dataset: HuggingFace DatasetDict with train/validation/test splits
        dataset_name: Name of the dataset (used for saving)
    
    Returns:
        Preprocessed DatasetDict
    """
    def preprocess_split(examples: Dict[str, list]) -> Dict[str, list]:
        """Preprocess a batch of examples."""
        examples['input_text'] = [
            preprocess_arabic_text(text) for text in examples['input_text']
        ]
        return examples

    # Create output directory
    output_dir = f"data/processed/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Process each split
    processed_dataset = DatasetDict()
    for split in dataset:
        print(f"\nProcessing {split} split...")
        
        # Apply preprocessing
        processed_split = dataset[split].map(
            preprocess_split,
            batched=True,
            desc=f"Preprocessing {split}"
        )
        
        # Save to disk
        output_file = f"{output_dir}/{split}.csv"
        df = pd.DataFrame(processed_split)
        df.to_csv(output_file, index=False)
        print(f"Saved {split} split to {output_file}")
        
        processed_dataset[split] = processed_split 
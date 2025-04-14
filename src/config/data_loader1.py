"""
📦 data_loader.py

This module handles loading and preparing the Arabic Reviews dataset
for sentiment analysis tasks.

Dataset: Arabic Reviews 100k
- Source: ar_reviews_100k.tsv
- Task: Classify sentiment of Arabic text (Positive, Negative, Mixed)
- Location: data/raw/ar_reviews/ar_reviews_100k.tsv

The loader function:
- Loads from local TSV
- Renames fields to: 'input_text' and 'label'
- Normalizes text and labels
- Splits into train / validation / test sets
- Returns a `datasets.DatasetDict` object

🧪 Example Usage:
    from src.config.data_loader1 import load_arabic_reviews
    dataset = load_arabic_reviews()
    print(dataset["train"][0])
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def show_sample(dataset, num_samples=5):
    """Display sample data from each split of the dataset."""
    for split in dataset.keys():
        print(f"\n{'-'*20} {split} {'-'*20}")
        samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}:")
            print(f"Text: {sample['input_text']}")
            print(f"Label: {sample['label']}\n")

def load_arabic_reviews(file_path="data/raw/ar_reviews_100k.tsv", 
                       val_size=0.1, 
                       test_size=0.1,
                       random_state=42):
    """
    Load and prepare the Arabic Reviews dataset for sentiment analysis.
    
    Args:
        file_path (str): Path to the TSV file
        val_size (float): Size of validation split (0-1)
        test_size (float): Size of test split (0-1)
        random_state (int): Random seed for reproducibility
    
    Returns:
        datasets.DatasetDict: Dataset with train/validation/test splits
    """
    print(f"Loading Arabic reviews from: {file_path}")
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, sep="\t", encoding=encoding)
                print(f"Successfully loaded data with encoding: {encoding}")
                print(f"Dataset shape: {df.shape}")
                print("\nFirst few rows:")
                print(df.head())
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading with {encoding}: {str(e)}")
                continue
    except Exception as e:
        print(f"Failed to load file: {str(e)}")
        return None

    # Normalize column names
    df = df.rename(columns={
        'text': 'input_text',
        'sentiment': 'label'
    })

    # Normalize labels to lowercase
    df['label'] = df['label'].str.strip().str.lower()

    # Create splits
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(val_size + test_size), random_state=random_state)

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df[['input_text', 'label']].reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df[['input_text', 'label']].reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df[['input_text', 'label']].reset_index(drop=True))
    })

    return dataset_dict

if __name__ == "__main__":
    # Load Arabic reviews dataset
    dataset = load_arabic_reviews()
    if dataset:
        print("\nArabic Reviews Dataset:")
        show_sample(dataset)
        
        # Preprocess and save dataset
        import sys
        sys.path.append('.')  # Add the project root to Python path
        from src.utils.preprocessing import preprocess_and_save
        preprocess_and_save(dataset, "ar_reviews")

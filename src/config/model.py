from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple

def build_model(model_name: str = "UBC-NLP/MARBERT", num_labels: int = 3) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Build a MARBERT model for sequence classification with the specified number of labels.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        num_labels (int): Number of output labels for classification
        
    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: Model and tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with sequence classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    return model, tokenizer 
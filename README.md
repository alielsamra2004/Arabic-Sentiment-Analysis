# Arabic Sentiment Analysis

A robust sentiment analysis system for Arabic text using MARBERT (Multilingual BERT), fine-tuned for sentiment classification. The system includes both a web interface and a command-line tool, supporting English-to-Arabic translation and sentiment prediction.

## Features

- 🔍 Sentiment analysis of Arabic text
- 🔄 English-to-Arabic translation support
- 📊 Probability visualization for sentiment predictions
- 💻 Both web interface and CLI tools
- 📝 Comprehensive logging of predictions
- 🎯 Fine-tuned MARBERT model

## Project Structure

```
arabic-sentiment-analysis/
├── .gradio/
│   └── certificate.pem
│
├── app/
│   └── interface.py      # Gradio web interface implementation
│
├── data/					# Processed and raw data
│   ├── processed/  
│   └── raw/
│
├── notebooks/
│   └── eda.ipynb			# Preprocessing and EDA of the data
│
├── outputs/
│   ├── ar_reviews/
│   │    └── best_model/       # Saved model files
│   │    └── confusion_matrix/       # Confusion matrix of best model
│   │    └── model_comparison/       # Comparing multiple models
│	└──  generate_confusion_matrix.py		# Generates the confusions matrix of the best model
│   └──  prediction_log.csv 		# Log of all predictions from the predict.py file using the terminal CLI
│   └──  predictions_log.csv		# Logs of predictions from the web interface
│
├── src/
│   └── config/
│       ├── data_loader1.py      # Script that loads the data 
│       ├── model.py      # Model architecture and configuration
│       ├── predict.py    # CLI prediction tool
│       └── train.py      # Model training script
│
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd arabic-sentiment-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Execution Guide

### Step 1: Terminal Setup
First, open your terminal and navigate to the project directory:
```bash
cd path/to/arabic-sentiment-analysis
```

Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Choose Your Interface

You have two options for using the system:

#### Option A: Command Line Interface (CLI)
Run the prediction script from the terminal:
```bash
# For English text (will be translated automatically)
python src/config/predict.py --text "The food was delicious and the service was excellent"

# For Arabic text (direct analysis)
python src/config/predict.py --text "الطعام كان لذيذا والخدمة كانت ممتازة"
```

The CLI tool will:
1. Process your input (translate if needed)
2. Show the Arabic text
3. Display sentiment analysis results
4. Show a probability visualization graph
5. Save results to `outputs/prediction_log.csv`

#### Option B: Web Interface
Launch the interactive web application:
```bash
python app/interface.py
```

Then:
1. Open your browser to the displayed URL (typically http://localhost:7860)
2. Use the interface to input text and see results in real-time

### File Execution Order

If you're working with the codebase, here's the recommended order for running/modifying files:

1. **Model Configuration** (if needed)
   - Edit `src/config/model.py` for model architecture changes
   - This file defines the MARBERT model structure

2. **Training** (if retraining is needed)
   - Run `python src/config/train.py`
   - This will create/update the model in `outputs/ar_reviews/best_model/`

3. **Prediction Tools**
   - CLI: `python src/config/predict.py --text "Your text here"`
   - Web: `python app/interface.py`

### Output Locations

After running either interface:
- Predictions are logged to: `outputs/prediction_log.csv`
- Visualizations are shown in real-time
- Model files are stored in: `outputs/ar_reviews/best_model/`

## Model Details

The system uses MARBERT (Multilingual Arabic BERT), fine-tuned for sentiment classification with three classes:
- Positive
- Negative
- Mixed/Neutral

The model processes Arabic text directly and includes translation capabilities for English input using the Helsinki-NLP translation model.

## Dependencies

Key dependencies include:
- transformers
- torch
- gradio
- matplotlib
- pandas
- arabic-reshaper
- python-bidi

See `requirements.txt` for complete list of dependencies.

## Logging

All predictions are automatically logged to `outputs/prediction_log.csv` with:
- Timestamp of prediction
- Input text (English)
- Arabic translation
- Predicted sentiment
- Probability distribution across sentiment classes

## 📚 Citations

If you use this project or build upon it, please consider citing the following works:

### Dataset: AR Reviews

ElSahar, Hady and El-Beltagy, Samhaa R.  
**Building Large Arabic Multi-domain Resources for Sentiment Analysis**  
In: *Computational Linguistics and Intelligent Text Processing*.  
Springer, 2015, pp. 23–34.  
[DOI: 10.1007/978-3-319-18117-2_2](https://doi.org/10.1007/978-3-319-18117-2_2)

```bibtex
@InProceedings{10.1007/978-3-319-18117-2_2,
  author    = {ElSahar, Hady and El-Beltagy, Samhaa R.},
  title     = {Building Large Arabic Multi-domain Resources for Sentiment Analysis},
  booktitle = {Computational Linguistics and Intelligent Text Processing},
  year      = {2015},
  publisher = {Springer International Publishing},
  pages     = {23--34},
  isbn      = {978-3-319-18117-2}
}
```

## Transformer Model: MARBERT

Abdul-Mageed, Muhammad; Elmadany, AbdelRahim; and Nagoudi, El Moatez Billah
**ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic**
In: *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021.*
[DOI: 10.18653/v1/2021.acl-long.551](https://doi.org/10.18653/v1/2021.acl-long.551)
[GitHub: UBC-NLP/MARBERT](https://github.com/UBC-NLP/MARBERT)

```bibtex
@inproceedings{abdul-mageed-etal-2021-arbert,
  title     = "{ARBERT} {\&} {MARBERT}: Deep Bidirectional Transformers for {A}rabic",
  author    = "Abdul-Mageed, Muhammad and Elmadany, AbdelRahim and Nagoudi, El Moatez Billah",
  booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
  year      = "2021",
  publisher = "Association for Computational Linguistics",
  address   = "Online",
  pages     = "7088--7105",
  url       = "https://aclanthology.org/2021.acl-long.551",
  doi       = "10.18653/v1/2021.acl-long.551"
}
```

## License
MIT

# Data Directory

## Overview
This directory contains the raw and processed versions of the Sentiment Labelled Sentences Dataset used for sentiment analysis with neural networks.

## Dataset Source
**Name**: Sentiment Labelled Sentences Data Set  
**Source**: UC Irvine Machine Learning Repository  
**Link**: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences  
**Paper**: Kotzias et al., "From Group to Individual Labels using Deep Features", KDD 2015

## Dataset Description
The dataset contains 3,000 sentences labeled with sentiment:
- **Amazon reviews**: 1,000 sentences (500 positive, 500 negative)
- **IMDB reviews**: 1,000 sentences (500 positive, 500 negative)
- **Yelp reviews**: 1,000 sentences (500 positive, 500 negative)

Each sentence is labeled as:
- `1` = Positive sentiment
- `0` = Negative sentiment

## Directory Structure

### raw/
Contains the original dataset files as downloaded:
- `amazon_cells_labelled.txt`
- `imdb_labelled.txt`
- `yelp_labelled.txt`

**Format**: Tab-separated values (sentence\tlabel)

**Example**:
```
Wow... Loved this place.	1
Not worth the time or money.	0
```

### processed/
Contains preprocessed and tokenized data ready for model training:

**Data files**:
- `train.csv` - Training set (80% of data, 2,400 samples)
- `test.csv` - Test set (20% of data, 600 samples)
- `eda_results.csv` - EDA analysis with original and cleaned text

**Model configuration**:
- `config.json` - Contains vocabulary size, max sequence length, and max words
- `tokenizer.json` - Keras tokenizer fitted on training data

**Numerical arrays** (NumPy format):
- `X_train.npy` - Training sequences (shape: [2400, 100])
- `X_test.npy` - Test sequences (shape: [600, 100])
- `y_train.npy` - Training labels (shape: [2400,])
- `y_test.npy` - Test labels (shape: [600,])

## Preprocessing Pipeline

The preprocessing pipeline is executed in `notebooks/01_eda.ipynb` and uses functions from `src/preprocessing.py`.

### 1. Text Cleaning
- **Lowercasing**: All text converted to lowercase
- **URL removal**: Removes http/https links
- **Email removal**: Removes email addresses
- **Mention removal**: Removes @mentions and #hashtags
- **Number removal**: Optional (disabled by default)
- **Punctuation removal**: Removes all punctuation marks
- **Whitespace normalization**: Removes extra spaces

### 2. Stopword Removal
- Uses NLTK English stopwords corpus
- Removes common words like "the", "is", "and", "a", etc.
- Reduces vocabulary size by ~47%
- Improves semantic signal

### 3. Tokenization and Encoding
- **Tokenizer**: Keras `Tokenizer` with vocabulary limit
- **Vocabulary size**: 10,000 most frequent words
- **Max sequence length**: 100 tokens
- **Padding**: Post-padding (adds zeros at the end)
- **Truncating**: Post-truncating (cuts from the end)
- **OOV token**: `<OOV>` for out-of-vocabulary words

### 4. Train-Test Split
- **Split ratio**: 80% training, 20% testing
- **Strategy**: Stratified split (maintains class balance)
- **Random seed**: 42 (for reproducibility)

## Data Statistics

### Overall Statistics (After Preprocessing)
- **Total samples**: 3,000
- **Positive samples**: 1,500 (50.0%)
- **Negative samples**: 1,500 (50.0%)
- **Vocabulary size (original)**: ~6,000 unique words
- **Vocabulary size (preprocessed)**: ~3,200 unique words
- **Vocabulary reduction**: ~47%
- **Average sentence length (original)**: ~15 words
- **Average sentence length (preprocessed)**: ~8 words
- **Length reduction**: ~47%

### Split Statistics
- **Training set**: 2,400 samples
  - Positive: 1,200 (50%)
  - Negative: 1,200 (50%)
- **Test set**: 600 samples
  - Positive: 300 (50%)
  - Negative: 300 (50%)

### Tokenizer Configuration
```json
{
  "vocab_size": 6145,
  "max_len": 100,
  "max_words": 10000
}
```

## Usage

### Loading Raw Data
```python
from src.preprocessing import load_sentiment_data

# Load all sources (Amazon, IMDb, Yelp)
df = load_sentiment_data(data_dir='data/raw')

# Load specific sources only
df = load_sentiment_data(data_dir='data/raw', sources=['amazon', 'imdb'])
```

### Loading Processed Data
```python
import pandas as pd
import numpy as np
import json

# Load DataFrames
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Load numerical arrays
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

# Load configuration
with open('data/processed/config.json', 'r') as f:
    config = json.load(f)

print(f"Vocabulary size: {config['vocab_size']}")
print(f"Max sequence length: {config['max_len']}")
```

### Loading Tokenizer
```python
from src.preprocessing import load_tokenizer_config

# Load tokenizer and config
tokenizer, config = load_tokenizer_config('data/processed')

# Use tokenizer for new texts
new_texts = ["This product is amazing!", "Terrible experience"]
sequences = tokenizer.texts_to_sequences(new_texts)
```

### Full Preprocessing Example
```python
from src.preprocessing import (
    load_sentiment_data,
    preprocess_texts,
    train_test_split,
    create_tokenizer,
    texts_to_sequences
)

# 1. Load data
df = load_sentiment_data(data_dir='data/raw')

# 2. Preprocess texts
df['clean_text'] = preprocess_texts(
    df['sentence'].tolist(),
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True
)

# 3. Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Create tokenizer
tokenizer, vocab_size = create_tokenizer(
    train_df['clean_text'].tolist(),
    max_words=10000
)

# 5. Convert to sequences
X_train = texts_to_sequences(
    train_df['clean_text'].tolist(),
    tokenizer,
    max_len=100,
    max_words=10000
)
```

## Data Quality

### Validation Checks
- ✅ No missing values in original dataset
- ✅ All labels are binary (0 or 1)
- ✅ Perfect class balance (50-50 split)
- ✅ No duplicate sentences across sources
- ✅ UTF-8 encoding verified
- ✅ Stratified split maintains balance in train/test

### Preprocessing Effects
- **Vocabulary reduction**: ~47% (from ~6,000 to ~3,200 unique words)
- **Length reduction**: ~47% (from ~15 to ~8 words average)
- **Semantic signal**: Improved by removing stopwords
- **Class balance**: Maintained perfectly after all preprocessing

## File Sizes (Approximate)
```
data/
├── raw/                          (~500 KB)
│   ├── amazon_cells_labelled.txt    (~150 KB)
│   ├── imdb_labelled.txt            (~150 KB)
│   └── yelp_labelled.txt            (~150 KB)
│
└── processed/                    (~5 MB)
    ├── train.csv                    (~400 KB)
    ├── test.csv                     (~100 KB)
    ├── eda_results.csv              (~500 KB)
    ├── config.json                  (~1 KB)
    ├── tokenizer.json               (~500 KB)
    ├── X_train.npy                  (~2 MB)
    ├── X_test.npy                   (~500 KB)
    ├── y_train.npy                  (~20 KB)
    └── y_test.npy                   (~5 KB)
```

## Regenerating Processed Data

To regenerate all processed files from scratch:

1. Ensure raw data files are in `data/raw/`
2. Run the EDA notebook:
```bash
   jupyter notebook notebooks/01_eda.ipynb
```
3. Execute all cells in the notebook
4. Verify files in `data/processed/`

The EDA notebook (`01_eda.ipynb`) performs:
- Exploratory Data Analysis with visualizations
- Text preprocessing and cleaning
- Train-test split
- Tokenizer creation and fitting
- Sequence conversion and padding
- Saving all processed files

## Citation

If using this dataset, please cite:
```bibtex
@inproceedings{kotzias2015group,
  title={From group to individual labels using deep features},
  author={Kotzias, Dimitrios and Denil, Misha and De Freitas, Nando and Smyth, Padhraic},
  booktitle={Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={597--606},
  year={2015}
}
```

## Notes

- **Reproducibility**: Always use random seed 42 for splitting
- **Preprocessing order**: EDA must be run before training models
- **Tokenizer dependency**: Models require the saved tokenizer for inference
- **Memory considerations**: Numerical arrays use int32 format to save space
- **Class balance**: Perfect 50-50 split maintained throughout all processing steps
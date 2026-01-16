import re
import string
import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow import keras

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    print("NLTK stopwords no disponibles, creando set vacío. Error:", e)
    STOPWORDS = set()


def clean_text(text, lowercase=True, remove_punctuation=True, remove_numbers=False):
    if not isinstance(text, str):
        return ""
    
    if lowercase:
        text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = ' '.join(text.split())
    
    return text.strip()


def remove_stopwords_from_text(text):
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    return ' '.join(filtered_words)


def preprocess_text(text, lowercase=True, remove_punctuation=True, 
                   remove_stopwords=False, remove_numbers=False):
    text = clean_text(text, lowercase, remove_punctuation, remove_numbers)
    if remove_stopwords:
        text = remove_stopwords_from_text(text)
    return text


def preprocess_texts(texts, lowercase=True, remove_punctuation=True, 
                    remove_stopwords=False, remove_numbers=False):
    return [preprocess_text(t, lowercase, remove_punctuation, 
                           remove_stopwords, remove_numbers) for t in texts]


def create_tokenizer(texts, max_words=10000):
    tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=max_words,
        oov_token='<OOV>',
        lower=True
    )
    tokenizer.fit_on_texts(texts)
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    print(f"Tokenizer fitted. Vocabulary size (to be used in Embedding input_dim): {vocab_size}")
    return tokenizer, vocab_size


def texts_to_sequences(texts, tokenizer, max_len=100, max_words=None):
    sequences = tokenizer.texts_to_sequences(texts)

    if max_words is not None:
        oov_idx = tokenizer.word_index.get(tokenizer.oov_token, 1)
        capped_sequences = []
        for seq in sequences:
            capped = [w if (w == oov_idx or (w is not None and w < max_words)) else oov_idx for w in seq]
            capped_sequences.append(capped)
        sequences = capped_sequences

    padded = keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len, padding='post', truncating='post'
    )
    return np.array(padded, dtype=np.int32)


def load_sentiment_data(data_dir='data/raw', sources=None):
    if sources is None:
        sources = ['amazon', 'imdb', 'yelp']
    
    file_mapping = {
        'amazon': 'amazon_cells_labelled.txt',
        'imdb': 'imdb_labelled.txt',
        'yelp': 'yelp_labelled.txt'
    }
    
    all_data = []
    
    for source in sources:
        if source not in file_mapping:
            print(f"Warning: Unknown source '{source}'. Skipping.")
            continue
        
        file_path = f"{data_dir}/{file_mapping[source]}"
        
        try:
            sentences = []
            labels = []
            skipped = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        skipped += 1
                        continue
                    
                    parts = line.split('\t')
                    
                    if len(parts) != 2:
                        print(f"   Línea {line_num} formato incorrecto: {line[:50]}...")
                        skipped += 1
                        continue
                    
                    sentence, label = parts
                    
                    if label not in ['0', '1']:
                        print(f"  ⚠️ Línea {line_num} label inválida: '{label}'")
                        skipped += 1
                        continue
                    
                    sentences.append(sentence)
                    labels.append(int(label))
            
            df = pd.DataFrame({
                'sentence': sentences,
                'label': labels,
                'source': source
            })
            
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {source}", end="")
            if skipped > 0:
                print(f" (skipped {skipped} lines)")
            else:
                print()
                
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except Exception as e:
            print(f"Error loading {source}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded from any source!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples loaded: {len(combined_df)}")
    print(f"Positive samples: {sum(combined_df['label'] == 1)}")
    print(f"Negative samples: {sum(combined_df['label'] == 0)}")
    
    return combined_df


def train_test_split(df, test_size=0.2, random_state=42, stratify=True):
    from sklearn.model_selection import train_test_split as sklearn_split
    
    stratify_col = df['label'] if stratify else None
    
    train_df, test_df = sklearn_split(df, test_size=test_size, 
                                      random_state=random_state, 
                                      stratify=stratify_col)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def get_text_statistics(texts):
    lengths = [len(text.split()) for text in texts]
    char_lengths = [len(text) for text in texts]
    
    stats = {
        'num_texts': len(texts),
        'avg_word_length': np.mean(lengths),
        'std_word_length': np.std(lengths),
        'min_word_length': np.min(lengths),
        'max_word_length': np.max(lengths),
        'median_word_length': np.median(lengths),
        'avg_char_length': np.mean(char_lengths),
        'total_words': sum(lengths)
    }
    
    return stats


def save_processed_data(train_df, test_df, output_dir='data/processed'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    print(f"Processed data saved to {output_dir}")


def save_tokenizer_config(tokenizer, max_words, max_len, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'max_words': max_words,
        'max_len': max_len,
        'vocab_size': min(len(tokenizer.word_index) + 1, max_words)
    }
    
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    tokenizer_json = tokenizer.to_json()
    with open(f'{output_dir}/tokenizer.json', 'w') as f:
        f.write(tokenizer_json)
    
    print(f"Tokenizer config saved to {output_dir}")


def load_tokenizer_config(tokenizer_dir):
    with open(f'{tokenizer_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    with open(f'{tokenizer_dir}/tokenizer.json', 'r') as f:
        tokenizer_json = f.read()
    
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    print(f"Tokenizer loaded from {tokenizer_dir}")
    
    return tokenizer, config
def get_word_freq(sentences, n=20):
    from collections import Counter
    words = []
    for sentence in sentences:
        words.extend(sentence.lower().split())
    return Counter(words).most_common(n)


def get_vocab_size(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence.split())
    return len(vocab)
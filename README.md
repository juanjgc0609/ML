# IntegrativeTask2-TuringSentiment

## Sentiment Analysis with Neural Networks
*Course*: Computability and Complexity Theory  
*Institution*: ICESI University  
*Project Deadline*: November 17th, 2025

### Team Members
- Juan José Gordillo - juanjgc.0609@gmail.com - @juanjgc0609
- Anderson Olave - anderson20dj@gmail.com - @OI22A

---

## Project Overview

This project implements and compares multiple neural network architectures for sentiment analysis on the UCI Sentiment Labelled Sentences Dataset. We explore Dense Neural Networks, vanilla RNNs, LSTM networks, and Transformer-based models, connecting these modern architectures with classical Turing Machine concepts of memory, computation, and sequence processing.

### Objectives
- Implement and compare multiple neural architectures for sentiment classification
- Analyze the relationship between model complexity and performance
- Connect neural network architectures to Turing Machine concepts
- Evaluate the impact of memory mechanisms (LSTM gates) on sequence processing
- Assess the effectiveness of attention mechanisms in Transformers

---

## Project Structure


IntegrativeTask2-TuringSentiment/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                        # Original UCI dataset
│   ├── processed/                  # Preprocessed and tokenized data
│   │   ├── X_train.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   ├── y_test.npy
│   │   └── config.json
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb     # DummyClassifier baseline
│   ├── 03_dense_rnn_lstm.ipynb     # Dense NN, RNN, and LSTM models
│   ├── 04_transformer_extension.ipynb  # Transformer (DistilBERT) implementation
│   └── utils.ipynb                 # Helper functions
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Text cleaning, tokenization
│   ├── models.py                   # Model architectures
│   ├── train.py                    # Training routines
│   ├── evaluate.py                 # Evaluation metrics
│   └── visualize.py                # Visualization utilities
│
├── outputs/
│   ├── figures/                    # Training curves, comparisons
│   ├── metrics/                    # Performance metrics (JSON)
│   │   ├── baseline_summary.json
│   │   ├── Dense_NN_history.json
│   │   ├── Vanilla_RNN_history.json
│   │   └── LSTM_history.json
│   └── saved_models/               # Trained model checkpoints
│       ├── Dense_NN_best.keras
│       ├── Vanilla_RNN_best.keras
│       └── LSTM_best.keras
│
├── docs/
│   ├── report.pdf                  # Final technical report
│   ├── presentation.pdf            # Project presentation slides
│   └── references.bib              # Bibliography
│
├── prompts/
│   ├── prompt_logs.txt             # AI tool usage logs
│   └── AIGen_Interactions.md       # Detailed AI interaction records
│
└── logs/
    ├── training_logs.txt           # Complete training session logs
    └── experiment_notes.md         # Experimental observations and insights


---

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager
- CUDA-compatible GPU (optional, for faster training)

### Installation

bash
# Clone the repository
git clone https://github.com/your-username/IntegrativeTask2-TuringSentiment.git
cd IntegrativeTask2-TuringSentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"


### Dataset Setup
1. Download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)
2. Extract the files into data/raw/
3. The dataset contains three files:
   - amazon_cells_labelled.txt
   - imdb_labelled.txt
   - yelp_labelled.txt

---

## Running the Project

### Step 1: Exploratory Data Analysis
bash
jupyter notebook notebooks/01_eda.ipynb

- Analyze class distribution (balanced: 50% positive, 50% negative)
- Examine sentence length statistics
- Visualize vocabulary characteristics

### Step 2: Baseline Model
bash
jupyter notebook notebooks/02_baseline_model.ipynb

- Establish performance baseline with DummyClassifier
- Target to beat: F1-Score = 0.6667

### Step 3: Neural Models (Dense, RNN, LSTM)
bash
jupyter notebook notebooks/03_dense_rnn_lstm.ipynb

- Hyperparameter tuning with GridSearchCV (3-fold CV)
- Train Dense NN, Vanilla RNN, and LSTM models
- Compare performance metrics and training dynamics

### Step 4: Transformer Extension
bash
jupyter notebook notebooks/04_transformer_extension.ipynb

- Fine-tune DistilBERT on sentiment classification
- Compare with traditional neural architectures
- Analyze attention mechanisms

### Alternative: Run from scripts
bash
# Preprocess data
python src/preprocessing.py

# Train models
python src/train.py --model dense
python src/train.py --model rnn
python src/train.py --model lstm

# Evaluate models
python src/evaluate.py


---

## Models Implemented

### 1. Baseline
- *DummyClassifier*: Majority class strategy
- *Purpose*: Minimum performance threshold
- *F1-Score*: 0.6667

### 2. Dense Neural Network
- *Architecture*: Embedding → GlobalAveragePooling → Dense(128) → Dense(64) → Dense(32) → Output
- *Parameters*: 617,729 (2.36 MB)
- *Regularization*: Dropout (0.3), L2 (0.0005)
- *Training Time*: ~2-3 minutes
- *Test F1-Score*: 0.8145

### 3. Vanilla RNN
- *Architecture*: Embedding → SimpleRNN(64) → Dropout → Output
- *Parameters*: 605,313 (2.31 MB)
- *Regularization*: Dropout (0.3), L2 (0.0005)
- *Training Time*: ~1-2 minutes
- *Test F1-Score*: 0.7195
- *Issue*: Severe overfitting observed

### 4. LSTM Network
- *Architecture*: Embedding → Bidirectional LSTM(128) → Dropout(0.5) → Output
- *Parameters*: 856,321 (3.27 MB)
- *Regularization*: Dropout (0.5), Recurrent Dropout (0.2), L2 (0.001)
- *Training Time*: ~45 minutes
- *Test F1-Score*: 0.7987

### 5. Transformer (DistilBERT)
- *Architecture*: distilbert-base-uncased (pre-trained)
- *Parameters*: ~67M
- *Fine-tuning*: 5 epochs, batch size 16, learning rate 2e-5
- *Training Time*: ~18 minutes
- *Test F1-Score*: 0.9268
- *Best Performance*: Epoch 3 (Val Acc: 0.9319)

---

## Results Summary

### Final Model Rankings

| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa | Parameters | Training Time |
|-------|----------|-----------|--------|----------|---------------|------------|---------------|
| *Transformer* | *0.9267* | *0.9270* | *0.9267* | *0.9268* | *0.8533* | 67M | ~18 min |
| *Dense NN* | *0.8033* | *0.8078* | *0.8033* | *0.8145* | *0.6067* | 617K | ~3 min |
| *LSTM* | *0.7950* | *0.7954* | *0.7950* | *0.7987* | *0.5900* | 856K | ~45 min |
| *Vanilla RNN* | *0.7167* | *0.7168* | *0.7167* | *0.7195* | *0.4333* | 605K | ~2 min |
| *Baseline* | *0.5500* | *0.5000* | *0.5500* | *0.6667* | *0.0000* | - | <1 min |

### Key Findings

1. *Transformer Dominance*: DistilBERT achieved 92.68% F1-score, outperforming all traditional models by 11+ percentage points
2. *Dense NN Surprise*: Simple Dense NN outperformed LSTM among traditional architectures (81.45% vs 79.87%)
3. *LSTM vs RNN*: Bidirectional LSTM with gating mechanisms significantly outperformed vanilla RNN (79.87% vs 71.95%)
4. *Vanilla RNN Limitations*: Severe overfitting (94% train, 67% validation) due to vanishing gradient problems
5. *Efficiency Trade-off*: Dense NN offers best performance-to-speed ratio among traditional models

### Performance Improvements over Baseline

- Transformer: +26.01 percentage points (39% relative improvement)
- Dense NN: +14.78 percentage points (22% relative improvement)
- LSTM: +13.20 percentage points (20% relative improvement)
- Vanilla RNN: +5.28 percentage points (8% relative improvement)

### Error Analysis

| Model | Total Errors | False Positives | False Negatives | Error Rate | Avg Error Confidence |
|-------|--------------|-----------------|-----------------|------------|---------------------|
| Dense NN | 118 | 77 (65.3%) | 41 (34.7%) | 19.67% | 0.2094 |
| Vanilla RNN | 170 | 88 (51.8%) | 82 (48.2%) | 28.33% | 0.2680 |
| LSTM | 123 | 67 (54.5%) | 56 (45.5%) | 20.50% | 0.2166 |

---

## Connection to Turing Machines

This project explores the relationship between classical Turing Machine concepts and modern neural architectures:

### Memory Mechanisms
- *Dense NN*: Fixed-size memory through embedding averaging (no sequential memory)
- *Vanilla RNN*: Sequential memory with limited capacity (vanishing gradients)
- *LSTM*: Selective memory with read/write gates (analogous to TM tape operations)
- *Transformer*: Attention-based memory with O(1) access to any position

### Computational Models
- *Sequential Processing*: RNN/LSTM process tokens one-by-one like TM tape reading
- *Parallel Processing*: Dense NN and Transformers process entire sequences simultaneously
- *State Transitions*: RNN hidden states analogous to TM state changes
- *Gating Mechanisms*: LSTM gates simulate conditional read/write operations

### Computational Power
- *Turing Completeness*: RNNs theoretically Turing complete with infinite precision
- *Practical Limitations*: Gradient issues restrict vanilla RNN effective memory span
- *Memory Extensions*: LSTM gates extend practical computational capacity
- *Attention Superiority*: Transformers provide non-sequential access unlike linear TM tape

---

## Hyperparameter Tuning

### GridSearchCV Configuration
- *Cross-Validation*: 3-fold stratified
- *Scoring Metric*: Accuracy
- *Search Strategy*: Exhaustive grid search

### Best Configurations Found

*Dense Neural Network* (12 combinations, 84 seconds)
- Dense units: [128, 64, 32]
- Dropout: 0.3
- L2 regularization: 0.0005
- CV Accuracy: 0.6583

*Vanilla RNN* (8 combinations, 76 seconds)
- RNN units: 64
- Dropout: 0.3
- L2 regularization: 0.0005
- CV Accuracy: 0.6412

*LSTM Network* (32 combinations, 2435 seconds)
- LSTM units: 128
- Bidirectional: True
- Dropout: 0.5
- Recurrent dropout: 0.2
- L2 regularization: 0.001
- CV Accuracy: 0.7646

---

## Documentation

- *Report*: See docs/report.pdf for comprehensive methodology, results, and analysis (2-4 pages)
- *Presentation*: See docs/presentation.pdf for 10-minute technical presentation
- *Training Logs*: See logs/training_logs.txt for complete training session records
- *Experiment Notes*: See logs/experiment_notes.md for detailed observations and insights
- *AI Usage*: See prompts/ for complete GenAI interaction logs

---

## AI Tool Usage

This project was developed with collaboration from Generative AI tools following Level 3 guidelines:
- AI assisted with code scaffolding, debugging, and documentation
- All AI-generated content was critically reviewed, modified, and validated by team members
- Complete interaction logs available in prompts/ directory demonstrate iterative refinement
- Final implementations showcase conceptual understanding and technical mastery
- Team members can explain and modify all code independently

---

## Team Contributions

- *Juan José Gordillo*: Data preprocessing pipeline, EDA implementation, baseline model setup, hyperparameter tuning framework, Dense NN implementation
- *Anderson Olave*: Vanilla RNN and LSTM implementation, training infrastructure, Transformer extension, evaluation metrics, visualization tools, error analysis, documentation and report writing

Both team members contributed equally to results analysis, project documentation, and presentation preparation.

---

## Repository Statistics

- *Total Commits*: 10+ (with 1-hour minimum intervals)
- *Languages*: Python (Jupyter Notebooks, Scripts)
- *Frameworks*: TensorFlow/Keras, Hugging Face Transformers, scikit-learn
- *Dataset Size*: 3,000 sentences (Amazon, IMDB, Yelp)
- *Train/Test Split*: 2,400 / 600 samples (80/20)

---

## Reproducibility

All experiments are fully reproducible:
1. Random seeds set for numpy and tensorflow
2. Configuration files saved in data/processed/config.json
3. Best model checkpoints preserved in outputs/saved_models/
4. Complete training history logged in JSON format
5. Hyperparameter search results documented

To reproduce results:
bash
# Run all notebooks in sequence
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_baseline_model.ipynb
jupyter notebook notebooks/03_dense_rnn_lstm.ipynb
jupyter notebook notebooks/04_transformer_extension.ipynb


---

## References

- Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015). From Group to Individual Labels using Deep Features. KDD.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. arXiv preprint arXiv:1910.01108.

Complete bibliography available in docs/references.bib

---

## Contact

For questions or issues, please contact any team member:
- Juan José Gordillo - juanjgc.0609@gmail.com
- Anderson Olave - anderson20dj@gmail.com

---

## License

This project is developed for academic purposes as part of the Computability and Complexity Theory course at ICESI University.

---

*Last Updated*: November 2024  
*Project Status*: Completed  
*Final Models*: Dense NN, Vanilla RNN, LSTM, Transformer (DistilBERT)  
*Best Performance*: 92.68% F1-Score (Transformer)
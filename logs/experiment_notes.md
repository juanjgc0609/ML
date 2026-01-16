# Experiment Notes - Sentiment Analysis Project

## Project Overview
**Objective**: Build and compare sentiment analysis models using Dense NN, Vanilla RNN, LSTM, and Transformer architectures  
**Dataset**: UCI Sentiment Labelled Sentences (Amazon, IMDB, Yelp)  
**Date**: November 2024  
**Team Members**: Juan Jose Gordillo, Anderson Olave

---

## Experimental Setup

### Dataset Characteristics
- **Total samples**: 3,000 sentences (1,000 per source)
- **Train/Test split**: 80/20 (2,400 / 600)
- **Class distribution**: Balanced (50% positive, 50% negative)
- **Vocabulary size**: 4,632 unique tokens
- **Sequence length**: Fixed at 100 tokens (padded/truncated)
- **Max words considered**: 10,000

### Preprocessing Pipeline
1. Text lowercasing
2. Tokenization using Keras Tokenizer
3. Sequence padding/truncation to length 100
4. Train-test split with stratification
5. Configuration saved for reproducibility

---

## Experimental Design

### Phase 1: Baseline Model
**Model**: DummyClassifier (most frequent strategy)  
**Purpose**: Establish minimum performance threshold  
**Result**: F1-Score = 0.6667

**Key Insight**: Any model below this threshold is worse than random guessing with class distribution knowledge.

---

### Phase 2: Hyperparameter Tuning

#### Approach
- Method: GridSearchCV with 3-fold cross-validation
- Scoring metric: Accuracy
- Strategy: Exhaustive search over parameter grid

#### Dense Neural Network
**Search space**:
- Dense units: [128, 64], [256, 128], [128, 64, 32]
- Dropout rate: 0.3, 0.5
- L2 regularization: 0.001, 0.0005

**Best configuration**: 
- Units: [128, 64, 32] (3 hidden layers)
- Dropout: 0.3
- L2: 0.0005
- CV Accuracy: 0.6583

**Observations**:
- Deeper architecture (3 layers) performed best
- Lower dropout (0.3) preferred over 0.5
- Moderate L2 regularization optimal

#### Vanilla RNN
**Search space**:
- RNN units: 64, 128
- Dropout rate: 0.3, 0.5
- L2 regularization: 0.001, 0.0005

**Best configuration**:
- Units: 64
- Dropout: 0.3
- L2: 0.0005
- CV Accuracy: 0.6412

**Observations**:
- Smaller RNN (64 units) outperformed larger (128)
- Suggests capacity issues with vanilla RNN
- Prone to overfitting despite regularization

#### LSTM Network
**Search space**:
- LSTM units: 64, 128
- Bidirectional: True, False
- Dropout rate: 0.3, 0.5
- Recurrent dropout: 0.0, 0.2
- L2 regularization: 0.001, 0.0005

**Best configuration**:
- Units: 128
- Bidirectional: True
- Dropout: 0.5
- Recurrent dropout: 0.2
- L2: 0.001
- CV Accuracy: 0.7646

**Observations**:
- Bidirectionality crucial for performance
- Higher dropout (0.5) beneficial for LSTM
- Recurrent dropout helps prevent overfitting
- Strong CV performance (16% better than Dense NN)

---

### Phase 3: Final Training

#### Training Configuration
```python
epochs = 50
batch_size = 32
validation_split = 0.2
optimizer = Adam(lr=0.001)
loss = 'binary_crossentropy'
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint(save_best_only=True)
]
```

#### Dense Neural Network Results
- **Training time**: ~2-3 minutes
- **Best epoch**: Variable (early stopping)
- **Validation accuracy**: ~0.75-0.80
- **Convergence**: Smooth, gradual improvement
- **Overfitting**: Minimal, good train-val consistency

**Architecture effectiveness**:
- GlobalAveragePooling simplified sequence representation
- 3-layer structure captured adequate complexity
- Dropout effectively prevented overfitting

#### Vanilla RNN Results
- **Training time**: ~1-2 minutes
- **Best epoch**: 3 (early stopping triggered)
- **Validation accuracy**: 0.6750
- **Convergence**: Rapid initial learning, then plateau
- **Overfitting**: SEVERE (train acc >0.94, val acc 0.67)

**Problems identified**:
- Vanishing gradient issues evident
- Failed to learn long-term dependencies
- Memorized training data without generalization
- Not suitable for this task despite being sequence-aware

#### LSTM Network Results
- **Training time**: ~7-11 minutes
- **Best epoch**: 3
- **Validation accuracy**: 0.7833
- **Convergence**: Fast, stable after epoch 3
- **Overfitting**: Moderate (train acc 0.96, val acc 0.78)

**Architecture effectiveness**:
- Bidirectional processing captured context well
- Gating mechanisms prevented vanishing gradients
- Recurrent dropout crucial for regularization
- Best validation performance achieved

---

## Test Set Performance Analysis

### Final Rankings
1. **Transformer (DistilBERT)**: F1 = 0.9268
2. **Dense NN**: F1 = 0.8145
3. **LSTM**: F1 = 0.7987
4. **Vanilla RNN**: F1 = 0.7195
5. **Baseline**: F1 = 0.6667

### Dense NN Outperforming LSTM (without Transformer)

**Possible explanations**:
1. **Dataset simplicity**: Sentiment often expressed in local n-grams, not long dependencies
2. **Sequence length**: 100 tokens may not require sequential processing
3. **Data volume**: 2,400 samples may be insufficient for complex LSTM
4. **Overfitting**: LSTM (856K params) vs Dense NN (617K params) - more parameters with limited data
5. **Embedding averaging**: GlobalAveragePooling may be ideal for this bag-of-sentiment-words task

### Error Analysis Insights

#### Dense NN Errors (118 total)
- **False Positives**: 77 (65.3%)
- **False Negatives**: 41 (34.7%)
- **Pattern**: Tends to over-predict positive sentiment
- **Confidence**: Low (0.2094) - uncertain when wrong

#### Vanilla RNN Errors (170 total)
- **False Positives**: 88 (51.8%)
- **False Negatives**: 82 (48.2%)
- **Pattern**: Balanced errors suggest random guessing when uncertain
- **Confidence**: Higher (0.2680) - dangerously confident when wrong

#### LSTM Errors (123 total)
- **False Positives**: 67 (54.5%)
- **False Negatives**: 56 (45.5%)
- **Pattern**: Slight positive bias
- **Confidence**: Low (0.2166) - appropriately uncertain

---

## Transformer Extension (Optional)

### Architecture: DistilBERT
Pre-trained model: distilbert-base-uncased
Total parameters: ~67M
Fine-tuning approach: Full model training
Sequence length: 100 tokens

### Training Configuration
Epochs: 5
Batch size: 16
Learning rate: 2e-5
Optimizer: AdamW
Warmup steps: 0

### Training Progress

**Epoch 1**
- Train Loss: 0.6175, Train Acc: 0.6500
- Val Loss: 0.2839, Val Acc: 0.8972
- Training time: ~4 minutes
- Observation: Rapid initial learning from pre-trained weights

**Epoch 2**
- Train Loss: 0.2468, Train Acc: 0.9065
- Val Loss: 0.2079, Val Acc: 0.9222
- Training time: ~3.5 minutes
- Observation: Continued improvement, no overfitting signs

**Epoch 3**
- Train Loss: 0.1730, Train Acc: 0.9345
- Val Loss: 0.2096, Val Acc: 0.9319
- Training time: ~3.5 minutes
- Observation: Best validation performance achieved

**Epoch 4**
- Train Loss: 0.1403, Train Acc: 0.9482
- Val Loss: 0.2179, Val Acc: 0.9278
- Training time: ~3 minutes
- Observation: Slight validation degradation, patience counter started

**Epoch 5**
- Train Loss: 0.1066, Train Acc: 0.9583
- Val Loss: 0.2221, Val Acc: 0.9250
- Training time: ~3 minutes
- Observation: Overfitting emerging, gap between train and validation widening

### Test Set Results
Accuracy: 0.9267
Precision: 0.9270 (macro avg)
Recall: 0.9267 (macro avg)
F1-Score: 0.9268
Cohen's Kappa: 0.8533

Class-wise Performance:
- Negative: Precision 0.9333, Recall 0.9200, F1 0.9266
- Positive: Precision 0.9207, Recall 0.9333, F1 0.9270

### Performance Comparison
Improvement over Dense NN: +11.23 percentage points
Improvement over LSTM: +12.81 percentage points
Improvement over baseline: +26.01 percentage points

### Key Observations

**Advantages of Transformer**
1. Pre-trained contextual embeddings capture semantic relationships
2. Self-attention mechanism models dependencies effectively
3. Bidirectional context understanding superior to LSTM
4. Minimal architecture engineering required
5. State-of-the-art performance with minimal fine-tuning

**Computational Trade-offs**
1. Training time: ~18 minutes total vs ~3 minutes for Dense NN
2. Model size: 67M parameters vs 617K for Dense NN (108x larger)
3. Inference time: Slower per sample
4. Memory requirements: Significantly higher

**Why Transformer Dominates**
1. Transfer learning from massive pre-training corpus
2. Attention weights learn task-specific features quickly
3. No vanishing gradient issues like vanilla RNN
4. Better than LSTM at capturing long-range dependencies
5. Superior to Dense NN through contextual understanding vs bag-of-words

---

## Key Findings and Lessons Learned

### 1. Architecture Selection
Dense NN proved most effective among traditional models
Vanilla RNN not viable due to severe overfitting and gradient issues
LSTM competitive but didn't justify added complexity
Transformer architecture achieves state-of-the-art performance when available

### 2. Hyperparameter Impact
- **Dropout**: Critical for all models (optimal varied by architecture)
- **L2 regularization**: Consistent benefit across all models
- **Bidirectionality**: Massive improvement for LSTM (+10% accuracy)
- **Network depth**: 3 layers optimal for Dense NN

### 3. Training Observations
- **Early stopping essential**: Prevented overfitting in all models
- **Learning rate scheduling**: ReduceLROnPlateau helped fine-tune
- **Batch size 32**: Good balance of speed and stability

### 4. Task-Specific Insights
- Sentiment analysis may not require sequential processing
- Local features (words, short phrases) more important than long-range dependencies
- Average pooling of embeddings surprisingly effective

### 5. Computational Efficiency
- Dense NN: Fastest training, best performance among traditional models
- Vanilla RNN: Fast but poor results
- LSTM: Moderate speed (~4x Dense NN) with slightly worse results
- Transformer: Slowest (~6x Dense NN) but best overall performance

---

## Comparison with Literature

### Expected vs Actual Results

**Expected** (based on typical NLP benchmarks):
- LSTM should outperform Dense NN
- Sequential models should excel at text
- Transformers should dominate all traditional architectures

**Actual**:
- Dense NN outperformed LSTM among traditional models
- Sequential processing showed limited benefit for this task
- Transformer achieved expected state-of-the-art performance
- GlobalAveragePooling was surprisingly effective for bag-of-words approach

### Possible Dataset-Specific Factors
1. **Short sentences**: Average length may not require long-term memory
2. **Clear sentiment markers**: Keywords like "great", "terrible" are local features
3. **Balanced data**: No need for complex pattern recognition
4. **Limited training data**: Simpler models generalize better

---

## Relation to Turing Machine Concepts

### Memory and Computation
- Dense NN: Fixed-size memory (embedding average), parallel computation
- Vanilla RNN: Sequential memory with limited capacity (vanishing gradients)
- LSTM: Selective memory with gates (simulates Turing Machine tape operations)
- Transformer: Attention-based memory access, parallel processing of entire sequence

### Sequence Processing
- Sequential access: RNN/LSTM process tokens one-by-one (like TM tape reading)
- Parallel access: Dense NN and Transformer process all tokens simultaneously
- Memory gates: LSTM gates analogous to TM read/write/move operations
- Attention mechanism: Non-sequential access to any position, unlike linear TM tape

### Computational Power
- Turing completeness: RNNs theoretically Turing complete with infinite precision
- Practical limits: Gradient issues limit vanilla RNN memory span
- LSTM improvements: Gating mechanisms extend effective memory
- Transformer capacity: Self-attention provides O(1) access to any position vs O(n) for sequential models

---

## Future Work Suggestions

### 1. Model Improvements
- Fine-tune larger transformers (BERT-base, RoBERTa)
- Experiment with different pre-trained models
- Implement ensemble of Dense NN + Transformer
- Explore distillation from Transformer to smaller models

### 2. Data Augmentation
- Synonym replacement
- Back-translation
- Paraphrasing with language models
- Increase training samples to 5000+

### 3. Advanced Analysis
- Attention visualization for interpretability
- Error analysis by sentence length and complexity
- Domain-specific performance (Amazon vs IMDB vs Yelp)
- Adversarial testing with negations and sarcasm

### 4. Optimization
- Quantization for faster inference
- Pruning for model compression
- Knowledge distillation from Transformer to Dense NN
- Mixed precision training

### 5. Cross-domain Evaluation
- Test on out-of-domain sentiment datasets
- Evaluate transfer learning capabilities
- Multi-task learning with related NLP tasks

---

## Experimental Decisions Log

### Decision 1: Sequence Length = 100
**Rationale**: Covers 95%+ of sentences in dataset  
**Alternative considered**: Dynamic padding  
**Outcome**: Good choice, minimal information loss

### Decision 2: Embedding Dimension = 128
**Rationale**: Standard size, balances capacity and efficiency  
**Alternative considered**: 256 (more capacity)  
**Outcome**: Adequate for vocabulary size of 4,632

### Decision 3: 3-Fold CV for Tuning
**Rationale**: Fast with sufficient reliability  
**Alternative considered**: 5-fold (more robust)  
**Outcome**: Good balance, consistent results

### Decision 4: Early Stopping Patience = 5
**Rationale**: Allows recovery from plateaus  
**Alternative considered**: 3 (faster training)  
**Outcome**: Optimal, prevented premature stopping

### Decision 5: Implement Transformer Extension
**Rationale**: Test state-of-the-art architecture, compare with traditional models
**Alternative considered**: Stop at LSTM
**Outcome**: Confirmed transformer superiority, 11% improvement over Dense NN

### Decision 6: Fine-tune vs Feature Extraction
**Rationale**: Full fine-tuning for best performance
**Alternative considered**: Freeze base layers, train only classifier
**Outcome**: Full fine-tuning justified, achieved 0.9268 F1-score

---

## Code Quality and Reproducibility

### Reproducibility Measures
Random seeds set in numpy, tensorflow
Configuration files saved (config.json)
Model checkpoints preserved
Training history logged
Hyperparameter search results documented
Transformer training logs with detailed epoch-by-epoch metrics

### Code Organization
Modular design (src/ modules)
Clear notebook structure (01-04 numbered sequence)
Consistent naming conventions
Comprehensive comments
Git version control with 10+ commits  

---

## Challenges Encountered

### 1. Vanilla RNN Overfitting
Problem: Training accuracy 94%, validation 67%
Solution: Increased regularization, reduced capacity
Result: Still poor, confirmed vanilla RNN inadequate for this task

### 2. LSTM Training Time
Problem: 40+ minutes for hyperparameter search
Solution: Reduced search space, used 3-fold CV instead of 5-fold
Result: Manageable tuning time, found optimal configuration

### 3. Transformer Resource Requirements
Problem: High memory consumption, slow training compared to other models
Solution: Reduced batch size to 16, gradient accumulation considered but not needed
Result: Successful training in 18 minutes total

### 4. Convergence Instability in Early Experiments
Problem: Some configurations showed erratic training curves
Solution: Learning rate reduction schedule, increased patience for early stopping
Result: Stable training achieved across all models

---

## Conclusion

This experiment successfully demonstrated:

1. Dense Neural Networks can outperform sequential models (LSTM, RNN) on short-text sentiment analysis
2. Vanilla RNNs are inadequate for modern NLP tasks due to gradient problems and overfitting
3. LSTM provides moderate improvement over vanilla RNN but doesn't justify complexity for this dataset
4. Transformer architecture (DistilBERT) achieves state-of-the-art performance with 92.68% F1-score
5. Pre-trained language models provide significant advantages through transfer learning
6. Proper hyperparameter tuning is crucial across all architectures
7. Task complexity and dataset size influence optimal architecture selection

Final Rankings:
- Transformer: 0.9268 F1 (best overall, 26% improvement over baseline)
- Dense NN: 0.8145 F1 (best traditional model, 22% improvement over baseline)
- LSTM: 0.7987 F1 (competitive, 20% improvement over baseline)
- Vanilla RNN: 0.7195 F1 (limited improvement over baseline)
- Baseline: 0.6667 F1

The Dense NN remains the most efficient traditional model, while Transformer provides superior performance when computational resources and pre-trained models are available. For production deployment on this task, the choice between Dense NN and Transformer depends on latency requirements, computational budget, and performance targets.

---

## References and Resources

- Dataset: Kotzias et al. (2015) "From Group to Individual Labels using Deep Features", KDD
- Framework: TensorFlow/Keras, Hugging Face Transformers
- Pre-trained Model: DistilBERT (distilbert-base-uncased)
- Hyperparameter Tuning: scikit-learn GridSearchCV
- Metrics: sklearn.metrics
- Visualization: matplotlib, seaborn


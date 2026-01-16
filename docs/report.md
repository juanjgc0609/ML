# Sentiment Analysis on Product Reviews

- Juan Jose Gordillo - A00407891
- Anderson Olave Ibargüen - A00408142
---

## INTRODUCTION

The proliferation of online reviews has made sentiment analysis a critical tool for understanding customer opinions. This project develops and compares machine learning models to classify product reviews as positive or negative using supervised learning techniques. We evaluate Dense Neural Networks, vanilla Recurrent Neural Networks, and Long Short-Term Memory networks to identify the most effective approach for sentiment classification on product reviews from multiple domains.

---

## OBJECTIVES

The objectives of this project are to preprocess and analyze a multi-domain review dataset, implement baseline models to establish minimum performance thresholds, develop and optimize neural network architectures including Dense NN, RNN, and LSTM models, evaluate all models using comprehensive performance metrics, and conduct comparative analysis to identify the best-performing architecture for sentiment analysis tasks.

---

## DATASET AND PREPROCESSING

The dataset used is the Sentiment Labelled Sentences Dataset from the UC Irvine Machine Learning Repository, containing 3,000 reviews from three sources: Amazon product reviews, IMDb movie reviews, and Yelp restaurant reviews. Each source contributes 500 positive and 500 negative reviews, resulting in a perfectly balanced dataset. The data was split into 2,400 training samples and 600 test samples using stratified sampling to maintain class balance.


The preprocessing pipeline transformed raw text into numerical sequences suitable for neural network input. Text cleaning involved converting all text to lowercase, removing URLs, email addresses, and punctuation, and normalizing whitespace. Common English stopwords were removed using the NLTK corpus to reduce noise while preserving semantic content. The Keras Tokenizer was employed with a maximum vocabulary size of 10,000 words, resulting in a final vocabulary of 4,632 unique words. All sequences were padded to a uniform length of 100 tokens using post-padding and post-truncation strategies.

The preprocessing achieved significant dimensionality reduction while enhancing semantic signal. Vocabulary was reduced from 8,015 to 5,277 unique words, representing a 34.2% reduction. Average text length decreased from 11.83 to 6.20 words per review, a 47.6% reduction that maintained semantic content. After preprocessing, the most frequent words directly correlated with sentiment, with negative reviews containing words like "bad", "dont", and "worst", while positive reviews featured "great", "good", and "best".
---

## MODEL ARCHITECTURES AND TRAINING

### Baseline: Dummy Classifier

Four baseline strategies were implemented to establish minimum performance thresholds: most frequent (always predicting the majority class), stratified (predicting based on training distribution), uniform (random prediction), and constant (always predicting positive). The constant strategy achieved the highest baseline F1-score of 0.6667 by maximizing recall at the expense of precision, serving as the benchmark that sophisticated models must exceed.

![Figure 4: Baseline model comparison shows the Constant strategy achieving the highest F1-score (0.6667) among all dummy classifiers, establishing the minimum performance threshold for neural network models.](/outputs/figures/baseline_comparison.png)

![Figure 5: Confusion matrices for all baseline strategies reveal that the Constant strategy achieves perfect recall by always predicting positive, while Stratified and Uniform show more balanced but random-like predictions.](/outputs/figures/baseline_confusion_matrices.png)

### Dense Neural Network

The Dense NN architecture employs a feedforward design with an embedding layer of 128 dimensions, global average pooling to aggregate sequential information, three dense layers with 128, 64, and 32 units respectively using ReLU activation, dropout regularization at 30%, L2 regularization with coefficient 0.0005, and a sigmoid output layer for binary classification. The model contains 617,729 trainable parameters.

Training utilized the Adam optimizer with learning rate 0.001, binary cross-entropy loss, batch size of 32, and maximum 50 epochs with early stopping. Hyperparameter tuning via GridSearchCV with 3-fold cross-validation explored 12 configurations, identifying the optimal three-layer architecture with moderate dropout and light regularization. The model trained in 9.04 seconds, converging at epoch 22 with validation accuracy of 80.21%.

![Figure 6: Dense NN training curves show gradual learning with best performance at epoch 22, demonstrating controlled overfitting through regularization and early stopping mechanisms.](/outputs/figures/Dense_NN_training_history.png)

### Vanilla RNN

The vanilla RNN processes sequences using a SimpleRNN layer with 64 units, maintaining hidden states to capture sequential dependencies. The architecture includes an embedding layer of 128 dimensions, the SimpleRNN layer with L2 regularization of 0.0005 on kernel and recurrent weights, 30% dropout, and a sigmoid output layer. The model contains 605,313 parameters.

Training configuration matched the Dense NN with Adam optimizer and binary cross-entropy loss. Hyperparameter tuning explored 8 configurations via GridSearchCV, determining that 64 units with moderate dropout performed best. The model trained rapidly in 7.40 seconds but converged very early at epoch 3, achieving validation accuracy of 67.50%. This early convergence suggested limited learning capacity, likely due to vanishing gradient problems inherent in simple recurrent architectures.

![Figure 7: Vanilla RNN training curves reveal early convergence at epoch 3 and diverging validation loss, indicating vanishing gradient problems and limited capacity to capture complex patterns.](/outputs/figures/Vanilla_RNN_training_history.png)

### LSTM Network

The LSTM architecture addresses vanishing gradient limitations through gating mechanisms and bidirectional processing. The model employs an embedding layer of 128 dimensions, a bidirectional LSTM layer with 128 units per direction, 50% dropout on input connections, 20% recurrent dropout, L2 regularization of 0.001, and a sigmoid output layer. The bidirectional structure processes sequences forward and backward simultaneously, capturing context from both directions. The model contains 856,321 parameters.

Extensive hyperparameter tuning explored 32 configurations over 2,435 seconds, with bidirectional processing proving critical to performance. The model achieved the highest cross-validation accuracy of 76.46% during tuning. Training required 73.02 seconds, with best performance at epoch 3 and validation accuracy of 78.33%. Despite its complexity, the LSTM demonstrated excellent convergence and generalization.

![Figure 8: LSTM training curves demonstrate rapid convergence to strong performance at epoch 3, with stable validation accuracy indicating excellent generalization despite model complexity.](/outputs/figures/LSTM_training_history.png)

![Figure 9: Training comparison across all models shows Dense NN requiring more epochs (22) for convergence while RNN and LSTM converge early (epoch 3), with distinct training dynamics reflecting architectural differences.](/outputs/figures/training_accuracy_comparison.png)

---

## PERFORMANCE EVALUATION

The following table summarizes test set performance across all models:

| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
|-------|----------|-----------|--------|----------|---------------|
| Baseline (Constant) | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.0000 |
| Baseline (Stratified) | 0.5200 | 0.5208 | 0.5000 | 0.5102 | 0.0400 |
| **Dense NN** | **0.8033** | 0.7708 | **0.8633** | **0.8145** | **0.6067** |
| Vanilla RNN | 0.7167 | 0.7124 | 0.7267 | 0.7195 | 0.4333 |
| LSTM | 0.7950 | 0.7846 | 0.8133 | 0.7987 | 0.5900 |

![Figure 10: Performance comparison across all metrics shows Dense NN achieving the highest scores in accuracy, recall, F1-score, and Cohen's Kappa, establishing it as the best-performing model.](/outputs/figures/metrics_comparison.png)

The Dense Neural Network achieved the highest performance across all key metrics. With 80.33% accuracy and 81.45% F1-score, it demonstrated excellent balance between precision (77.08%) and recall (86.33%). The Cohen's Kappa of 0.6067 indicates substantial agreement beyond chance. Error analysis revealed 118 total errors with a 19.67% error rate, consisting of 77 false positives and 41 false negatives. The model exhibited low confidence on errors (0.2094 average), suggesting appropriate uncertainty calibration.

![Figure 11: Dense NN confusion matrix shows 223 true negatives and 259 true positives, with an optimistic bias reflected in 77 false positives versus 41 false negatives.](/outputs/figures/Dense_NN_confusion_matrix.png)

The LSTM achieved strong second-place performance with 79.50% accuracy and 79.87% F1-score. Its bidirectional architecture provided robust context understanding, resulting in balanced precision (78.46%) and recall (81.33%). The model produced 123 errors with a 20.50% error rate, showing good generalization despite having 856,321 parameters. The training dynamics revealed rapid convergence to strong performance despite architectural complexity.

![Figure 12: LSTM confusion matrix displays 233 true negatives and 244 true positives with balanced error distribution (67 false positives, 56 false negatives).](/outputs/figures/LSTM_confusion_matrix.png)

The Vanilla RNN underperformed both advanced architectures with 71.67% accuracy and 71.95% F1-score. While achieving the fastest training time of 7.40 seconds, the model produced 170 errors with a 28.33% error rate. The balanced error distribution between false positives (88) and false negatives (82) suggests the model struggled to learn distinctive patterns. Early convergence at epoch 3 and limited validation accuracy indicated fundamental architectural limitations from vanishing gradients.

![Figure 13: Vanilla RNN confusion matrix reveals the highest error count among neural networks (170 errors) with balanced distribution, indicating difficulty in learning meaningful patterns.](/outputs/figures/Vanilla_RNN_confusion_matrix.png)

---

## COMPARATIVE ANALYSIS

All neural network models significantly exceeded the baseline performance of 66.67% F1-score. The Dense NN achieved a 22.17% improvement, the LSTM a 19.80% improvement, and the Vanilla RNN a 7.92% improvement. Both Dense NN and LSTM surpassed the target improvement threshold of 15%, demonstrating successful learning of meaningful sentiment patterns.

![Figure 14: Training loss comparison reveals distinct learning dynamics, with LSTM and RNN converging rapidly while Dense NN shows gradual improvement, and validation loss patterns indicating different generalization behaviors.](/poutputs/figures/training_loss_comparison.png)

The Dense NN's superior performance represents a noteworthy finding, as it outperformed the more sophisticated LSTM architecture by 1.58% F1-score while training 8 times faster (9.04 seconds versus 73.02 seconds). This success can be attributed to several factors. The dataset's characteristics, particularly the short average sequence length of 6.20 words after preprocessing, reduced the advantage of sequential processing. The Dense NN's three-layer architecture with global average pooling effectively captured sentiment through word presence rather than word order. Optimal hyperparameter configuration with moderate dropout (0.3) and light regularization (0.0005) provided sufficient model capacity without overfitting. The model's simplicity matched the data complexity, demonstrating that architectural sophistication should align with task requirements.

The LSTM's strong performance validated its design for capturing long-range dependencies and handling sequential information. Bidirectional processing proved critical, as non-bidirectional configurations achieved only 50.45% validation accuracy compared to 76.46% for bidirectional variants. The model's gating mechanisms successfully managed information flow, avoiding the vanishing gradient problems that limited the Vanilla RNN. However, the marginal performance difference from Dense NN (1.58% F1-score) did not justify the 8-fold increase in training time for this particular dataset.

The Vanilla RNN's underperformance highlighted the limitations of simple recurrent architectures. Despite processing sequences and maintaining hidden states, the model struggled with gradient flow during backpropagation through time. The early convergence and plateaued validation accuracy indicated that the simple recurrent mechanism could not capture the nuanced patterns necessary for accurate sentiment classification. This demonstrates that sequential processing alone is insufficient without proper memory management mechanisms.

Hyperparameter optimization proved essential for all models. GridSearchCV with 3-fold cross-validation systematically explored parameter spaces, revealing that architecture depth (three layers for Dense NN), bidirectional processing (for LSTM), and regularization strength significantly impacted performance. The Dense NN's optimal configuration with three progressive layers provided the right balance of capacity and generalization. For LSTM, bidirectional processing emerged as the single most important factor, more impactful than unit count or dropout rates.

From a practical deployment perspective, the Dense NN offers the best trade-off between performance and computational efficiency for short-text sentiment analysis. Its 80.33% accuracy with 9.04-second training time makes it suitable for real-time applications and resource-constrained environments. The LSTM remains valuable for scenarios requiring maximum accuracy despite higher computational costs, or when processing longer sequences where sequential relationships become more critical. The Vanilla RNN should be avoided for sentiment analysis tasks, as both Dense NN and LSTM provide superior performance with comparable or acceptable training times.

---

## MODEL ARCHITECTURES AND TRAINING

### Baseline: Dummy Classifier

Four baseline strategies were implemented to establish minimum performance thresholds: most frequent (always predicting the majority class), stratified (predicting based on training distribution), uniform (random prediction), and constant (always predicting positive). The constant strategy achieved the highest baseline F1-score of 0.6667 by maximizing recall at the expense of precision, serving as the benchmark that sophisticated models must exceed.

### Dense Neural Network

The Dense NN architecture employs a feedforward design with an embedding layer of 128 dimensions, global average pooling to aggregate sequential information, three dense layers with 128, 64, and 32 units respectively using ReLU activation, dropout regularization at 30%, L2 regularization with coefficient 0.0005, and a sigmoid output layer for binary classification. The model contains 617,729 trainable parameters.

Training utilized the Adam optimizer with learning rate 0.001, binary cross-entropy loss, batch size of 32, and maximum 50 epochs with early stopping. Hyperparameter tuning via GridSearchCV with 3-fold cross-validation explored 12 configurations, identifying the optimal three-layer architecture with moderate dropout and light regularization. The model trained in 9.04 seconds, converging at epoch 22 with validation accuracy of 80.21%.

### Vanilla RNN

The vanilla RNN processes sequences using a SimpleRNN layer with 64 units, maintaining hidden states to capture sequential dependencies. The architecture includes an embedding layer of 128 dimensions, the SimpleRNN layer with L2 regularization of 0.0005 on kernel and recurrent weights, 30% dropout, and a sigmoid output layer. The model contains 605,313 parameters.

Training configuration matched the Dense NN with Adam optimizer and binary cross-entropy loss. Hyperparameter tuning explored 8 configurations via GridSearchCV, determining that 64 units with moderate dropout performed best. The model trained rapidly in 7.40 seconds but converged very early at epoch 3, achieving validation accuracy of 67.50%. This early convergence suggested limited learning capacity, likely due to vanishing gradient problems inherent in simple recurrent architectures.

### LSTM Network

The LSTM architecture addresses vanishing gradient limitations through gating mechanisms and bidirectional processing. The model employs an embedding layer of 128 dimensions, a bidirectional LSTM layer with 128 units per direction, 50% dropout on input connections, 20% recurrent dropout, L2 regularization of 0.001, and a sigmoid output layer. The bidirectional structure processes sequences forward and backward simultaneously, capturing context from both directions. The model contains 856,321 parameters.

Extensive hyperparameter tuning explored 32 configurations over 2,435 seconds, with bidirectional processing proving critical to performance. The model achieved the highest cross-validation accuracy of 76.46% during tuning. Training required 73.02 seconds, with best performance at epoch 3 and validation accuracy of 78.33%. Despite its complexity, the LSTM demonstrated excellent convergence and generalization.

---

## PERFORMANCE EVALUATION

The following table summarizes test set performance across all models:

| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
|-------|----------|-----------|--------|----------|---------------|
| Baseline (Constant) | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.0000 |
| Baseline (Stratified) | 0.5200 | 0.5208 | 0.5000 | 0.5102 | 0.0400 |
| **Dense NN** | **0.8033** | 0.7708 | **0.8633** | **0.8145** | **0.6067** |
| Vanilla RNN | 0.7167 | 0.7124 | 0.7267 | 0.7195 | 0.4333 |
| LSTM | 0.7950 | 0.7846 | 0.8133 | 0.7987 | 0.5900 |

The Dense Neural Network achieved the highest performance across all key metrics. With 80.33% accuracy and 81.45% F1-score, it demonstrated excellent balance between precision (77.08%) and recall (86.33%). The Cohen's Kappa of 0.6067 indicates substantial agreement beyond chance. Error analysis revealed 118 total errors with a 19.67% error rate, consisting of 77 false positives and 41 false negatives. The model exhibited low confidence on errors (0.2094 average), suggesting appropriate uncertainty calibration.

The LSTM achieved strong second-place performance with 79.50% accuracy and 79.87% F1-score. Its bidirectional architecture provided robust context understanding, resulting in balanced precision (78.46%) and recall (81.33%). The model produced 123 errors with a 20.50% error rate, showing good generalization despite having 856,321 parameters. The training dynamics revealed rapid convergence to strong performance despite architectural complexity.

The Vanilla RNN underperformed both advanced architectures with 71.67% accuracy and 71.95% F1-score. While achieving the fastest training time of 7.40 seconds, the model produced 170 errors with a 28.33% error rate. The balanced error distribution between false positives (88) and false negatives (82) suggests the model struggled to learn distinctive patterns. Early convergence at epoch 3 and limited validation accuracy indicated fundamental architectural limitations from vanishing gradients.

---

## COMPARATIVE ANALYSIS

All neural network models significantly exceeded the baseline performance of 66.67% F1-score. The Dense NN achieved a 22.17% improvement, the LSTM a 19.80% improvement, and the Vanilla RNN a 7.92% improvement. Both Dense NN and LSTM surpassed the target improvement threshold of 15%, demonstrating successful learning of meaningful sentiment patterns.

The Dense NN's superior performance represents a noteworthy finding, as it outperformed the more sophisticated LSTM architecture by 1.58% F1-score while training 8 times faster (9.04 seconds versus 73.02 seconds). This success can be attributed to several factors. The dataset's characteristics, particularly the short average sequence length of 6.20 words after preprocessing, reduced the advantage of sequential processing. The Dense NN's three-layer architecture with global average pooling effectively captured sentiment through word presence rather than word order. Optimal hyperparameter configuration with moderate dropout (0.3) and light regularization (0.0005) provided sufficient model capacity without overfitting. The model's simplicity matched the data complexity, demonstrating that architectural sophistication should align with task requirements.

The LSTM's strong performance validated its design for capturing long-range dependencies and handling sequential information. Bidirectional processing proved critical, as non-bidirectional configurations achieved only 50.45% validation accuracy compared to 76.46% for bidirectional variants. The model's gating mechanisms successfully managed information flow, avoiding the vanishing gradient problems that limited the Vanilla RNN. However, the marginal performance difference from Dense NN (1.58% F1-score) did not justify the 8-fold increase in training time for this particular dataset.

The Vanilla RNN's underperformance highlighted the limitations of simple recurrent architectures. Despite processing sequences and maintaining hidden states, the model struggled with gradient flow during backpropagation through time. The early convergence and plateaued validation accuracy indicated that the simple recurrent mechanism could not capture the nuanced patterns necessary for accurate sentiment classification. This demonstrates that sequential processing alone is insufficient without proper memory management mechanisms.

Hyperparameter optimization proved essential for all models. GridSearchCV with 3-fold cross-validation systematically explored parameter spaces, revealing that architecture depth (three layers for Dense NN), bidirectional processing (for LSTM), and regularization strength significantly impacted performance. The Dense NN's optimal configuration with three progressive layers provided the right balance of capacity and generalization. For LSTM, bidirectional processing emerged as the single most important factor, more impactful than unit count or dropout rates.

From a practical deployment perspective, the Dense NN offers the best trade-off between performance and computational efficiency for short-text sentiment analysis. Its 80.33% accuracy with 9.04-second training time makes it suitable for real-time applications and resource-constrained environments. The LSTM remains valuable for scenarios requiring maximum accuracy despite higher computational costs, or when processing longer sequences where sequential relationships become more critical. The Vanilla RNN should be avoided for sentiment analysis tasks, as both Dense NN and LSTM provide superior performance with comparable or acceptable training times.

---

## CONCLUSION

This project successfully developed and evaluated neural network architectures for sentiment analysis on product reviews. The Dense Neural Network emerged as the best-performing model with 80.33% accuracy and 81.45% F1-score, surpassing the baseline by 22.17% and exceeding the target improvement of 15%. The LSTM achieved strong second-place performance with 79.50% accuracy and 19.80% improvement over baseline. The Vanilla RNN, while functional, significantly underperformed due to vanishing gradient limitations.

A key finding is that simpler architectures can outperform complex models when properly optimized and matched to data characteristics. The Dense NN's success with short, preprocessed sequences demonstrates the importance of aligning model complexity with task requirements. The extensive preprocessing pipeline, which reduced text length by 47.6% while enhancing semantic signal, proved critical to enabling all models to learn effectively.

Hyperparameter optimization through GridSearchCV significantly improved model performance, with architecture depth for Dense NN and bidirectional processing for LSTM emerging as critical factors. All models demonstrated appropriate uncertainty calibration with low confidence on errors, indicating robust learning rather than overconfident predictions.

---

## FUTURE WORK

Several directions could extend this research. Implementing transformer-based models such as DistilBERT or BERT would leverage pre-trained language representations and attention mechanisms, potentially improving accuracy by 5-15 percentage points. Ensemble methods combining Dense NN and LSTM predictions could provide more robust classification through complementary strengths. Expanding to multi-class sentiment classification with fine-grained labels or aspect-based sentiment analysis would enable more nuanced opinion understanding. Cross-domain transfer learning could improve generalization across different product categories. Finally, deploying optimized models with techniques such as quantization and ONNX conversion would enable real-time inference in production environments.

---

## REFERENCES

Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015). From Group to Individual Labels using Deep Features. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Chollet, F. et al. (2015). Keras. https://keras.io

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.

Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5-6), 602-610.

Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval, 2(1-2), 1-135.

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.


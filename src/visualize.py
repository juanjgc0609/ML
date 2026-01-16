import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_history(history, model_name, metrics=['loss', 'accuracy'], 
                          save_path=None):
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history if isinstance(history, dict) else history['history']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_key = metric
        val_key = f'val_{metric}'
        
        if train_key in history_dict:
            epochs = range(1, len(history_dict[train_key]) + 1)
            ax.plot(epochs, history_dict[train_key], 'b-', label='Train', linewidth=2)
            
            if val_key in history_dict:
                ax.plot(epochs, history_dict[val_key], 'r-', label='Validation', linewidth=2)
            
            ax.set_title(f'{model_name} - {metric.capitalize()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            if val_key in history_dict:
                if metric == 'loss':
                    best_epoch = np.argmin(history_dict[val_key])
                else:
                    best_epoch = np.argmax(history_dict[val_key])
                ax.axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.5, 
                          label=f'Best: epoch {best_epoch+1}')
                ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, normalize=False, 
                         save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title_suffix = ' (Normalized)'
    else:
        fmt = 'd'
        title_suffix = ''
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(f'{model_name} - Confusion Matrix{title_suffix}', 
             fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_metrics_comparison(comparison_df, save_path=None):

    metrics = [col for col in comparison_df.columns if col != 'Model']
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = comparison_df[metric].values
        models = comparison_df['Model'].values
        
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
    
    plt.show()



def plot_training_comparison(histories_dict, metric='accuracy', save_path=None):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    
    for idx, (model_name, history) in enumerate(histories_dict.items()):
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history if isinstance(history, dict) else history['history']
        
        train_key = metric
        val_key = f'val_{metric}'
        
        if train_key in history_dict:
            epochs = range(1, len(history_dict[train_key]) + 1)
            
            ax1.plot(epochs, history_dict[train_key], color=colors[idx], 
                    linewidth=2, label=model_name)
            
            if val_key in history_dict:
                ax2.plot(epochs, history_dict[val_key], color=colors[idx], 
                        linewidth=2, label=model_name)
    
    ax1.set_title(f'Training {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel(metric.capitalize(), fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f'Validation {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(metric.capitalize(), fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training comparison saved to: {save_path}")
    
    plt.show()


def plot_error_distribution(y_test, y_pred_proba, bins=20, save_path=None):
    y_pred = (y_pred_proba > 0.5).astype(int)
    correct = y_test == y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(y_pred_proba[correct], bins=bins, color='green', alpha=0.7, 
            edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    ax1.set_title('Correct Predictions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Prediction Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(y_pred_proba[~correct], bins=bins, color='red', alpha=0.7, 
            edgecolor='black')
    ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2)
    ax2.set_title('Incorrect Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Prediction Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution saved to: {save_path}")
    
    plt.show()


def create_full_report(models_results, comparison_df, y_test, 
                      output_dir='outputs/figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations for report...")
    
    print("  - Metrics comparison...")
    plot_metrics_comparison(
        comparison_df, 
        save_path=f'{output_dir}/metrics_comparison.png'
    )
    
    
    print("  - Confusion matrices...")
    for model_name, results in models_results.items():
        plot_confusion_matrix(
            y_test, 
            results['y_pred'], 
            model_name,
            save_path=f'{output_dir}/{model_name}_confusion_matrix.png'
        )
    
    print(f"\n All visualizations saved to: {output_dir}")

def plot_class_distribution(df, label_col='label', figsize=(10, 6), save_path=None):
    
    plt.figure(figsize=figsize)
    df[label_col].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
    plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment (0=Negative, 1=Positive)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    balance = df[label_col].value_counts(normalize=True) * 100
    print(f"\nClass balance:")
    print(f"Negative: {balance[0]:.2f}%")
    print(f"Positive: {balance[1]:.2f}%")


def plot_source_distribution(df, source_col='source', figsize=(10, 6), save_path=None):
    
    plt.figure(figsize=figsize)
    source_counts = df[source_col].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    source_counts.plot(kind='bar', color=colors)
    plt.title('Sample Distribution by Source', fontsize=14, fontweight='bold')
    plt.xlabel('Source')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_length_distribution(df, length_col='length', label_col='label', 
                            figsize=(15, 5), save_path=None):
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    df[length_col].hist(bins=30, ax=ax1, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax1.set_title('Text Length Distribution', fontweight='bold')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.grid(axis='y', alpha=0.3)
    
    df[df[label_col] == 0][length_col].hist(bins=20, alpha=0.6, label='Negative', 
                                             color='#FF6B6B', ax=ax2)
    df[df[label_col] == 1][length_col].hist(bins=20, alpha=0.6, label='Positive', 
                                             color='#4ECDC4', ax=ax2)
    ax2.set_title('Length Distribution by Sentiment', fontweight='bold')
    ax2.set_xlabel('Number of Words')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    print("\n=== LENGTH STATISTICS BY SENTIMENT ===")
    print("\nNegative:")
    print(df[df[label_col] == 0][length_col].describe())
    print("\nPositive:")
    print(df[df[label_col] == 1][length_col].describe())


def plot_length_boxplot(df, length_col='length', label_col='label', 
                        figsize=(10, 6), save_path=None):

    plt.figure(figsize=figsize)
    df.boxplot(column=length_col, by=label_col, grid=False, 
               patch_artist=True, showmeans=True)
    plt.suptitle('')
    plt.title('Text Length Distribution by Sentiment', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment (0=Negative, 1=Positive)')
    plt.ylabel('Number of Words')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_length_comparison(df, original_col='length', clean_col='clean_length',
                          label_col='label', figsize=(15, 10), save_path=None):

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    axes[0, 0].hist(df[original_col], bins=30, alpha=0.7, color='#FFE66D', 
                    edgecolor='black')
    axes[0, 0].set_title('Text Length - Original', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Words')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    axes[0, 1].hist(df[clean_col], bins=30, alpha=0.7, color='#4ECDC4', 
                    edgecolor='black')
    axes[0, 1].set_title('Text Length - Preprocessed', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Words')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    df[df[label_col] == 0][clean_col].hist(bins=20, alpha=0.6, 
                                             label='Negative', color='#FF6B6B', 
                                             ax=axes[1, 0])
    df[df[label_col] == 1][clean_col].hist(bins=20, alpha=0.6, 
                                             label='Positive', color='#4ECDC4', 
                                             ax=axes[1, 0])
    axes[1, 0].set_title('Length by Sentiment - Preprocessed', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Words')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    df.boxplot(column=clean_col, by=label_col, ax=axes[1, 1], 
               grid=False, patch_artist=True, showmeans=True)
    axes[1, 1].set_title('Length Distribution by Sentiment', fontweight='bold')
    axes[1, 1].set_xlabel('Sentiment (0=Negative, 1=Positive)')
    axes[1, 1].set_ylabel('Number of Words')
    plt.suptitle('')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_top_words(df, text_col='clean_sentence', label_col='label', 
                   top_n=10, figsize=(15, 6), save_path=None):
    from collections import Counter
    
    def get_word_freq(sentences, n=top_n):
        words = []
        for sentence in sentences:
            words.extend(sentence.lower().split())
        return Counter(words).most_common(n)
    
    neg_words = get_word_freq(df[df[label_col] == 0][text_col])
    pos_words = get_word_freq(df[df[label_col] == 1][text_col])
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    import pandas as pd
    neg_words_df = pd.DataFrame(neg_words, columns=['word', 'freq'])
    axes[0].barh(neg_words_df['word'], neg_words_df['freq'], color='#FF6B6B')
    axes[0].set_title(f'Top {top_n} Words - Negative Sentiment', fontweight='bold')
    axes[0].set_xlabel('Frequency')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    pos_words_df = pd.DataFrame(pos_words, columns=['word', 'freq'])
    axes[1].barh(pos_words_df['word'], pos_words_df['freq'], color='#4ECDC4')
    axes[1].set_title(f'Top {top_n} Words - Positive Sentiment', fontweight='bold')
    axes[1].set_xlabel('Frequency')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return neg_words, pos_words


def plot_vocab_comparison(vocab_original, vocab_clean, figsize=(10, 6), save_path=None):
    import pandas as pd
    
    fig, ax = plt.subplots(figsize=figsize)
    vocab_data = pd.DataFrame({
        'Type': ['Original', 'Preprocessed'],
        'Vocabulary': [vocab_original, vocab_clean]
    })
    ax.bar(vocab_data['Type'], vocab_data['Vocabulary'], 
           color=['#FFE66D', '#4ECDC4'], edgecolor='black')
    ax.set_title('Vocabulary Size Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Unique Words')
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(vocab_data['Vocabulary']):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    print(f"\n=== VOCABULARY ANALYSIS ===")
    print(f"Original vocabulary: {vocab_original} unique words")
    print(f"Preprocessed vocabulary: {vocab_clean} unique words")
    print(f"Reduction: {vocab_original - vocab_clean} words ({(1 - vocab_clean/vocab_original)*100:.1f}%)")
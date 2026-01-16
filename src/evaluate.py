import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, verbose=1):
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_pred_proba = y_pred_proba.flatten()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
    }

    if verbose:
        print("EVALUATION METRICS")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title():15s}: {value:.4f}")

    return metrics, y_pred, y_pred_proba


def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)


def get_classification_report(y_test, y_pred, verbose=1):

    report = classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Positive'],
        digits=4
    )
    
    if verbose:
        print("CLASSIFICATION REPORT")
        print(report)
    
    return report


def evaluate_all_models(models_dict, X_test, y_test):
    
    results = {}
    
    print("EVALUATING ALL MODELS")
    
    for model_name, model in models_dict.items():
        print(f"\n>>> Evaluating {model_name}...")
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, verbose=0)
        
        results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': get_confusion_matrix(y_test, y_pred),
            'classification_report': get_classification_report(y_test, y_pred, verbose=0)
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | "
              f"Kappa: {metrics['cohen_kappa']:.4f}")
    
    return results


def compare_models(results_dict, save_path=None):
    comparison_data = []
    
    for model_name, result in results_dict.items():
        row = {'Model': model_name}
        
        if 'metrics' in result:
            row.update(result['metrics'])
        else:
            row.update(result)
        
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    
    sort_col = 'accuracy' if 'accuracy' in df.columns else 'f1_score'
    df = df.sort_values(sort_col, ascending=False)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nComparison saved to: {save_path}")
    
    return df


def print_comparison_table(comparison_df):

    print("MODEL COMPARISON")
    print(comparison_df.to_string(index=False))


def save_evaluation_results(results_dict, output_dir='outputs/metrics'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, result in results_dict.items():
        metrics_path = os.path.join(output_dir, f'{model_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            metrics = result['metrics'].copy()
            for key, value in metrics.items():
                if isinstance(value, np.floating):
                    metrics[key] = float(value)
            json.dump(metrics, f, indent=2)
        
        report_path = os.path.join(output_dir, f'{model_name}_report.txt')
        with open(report_path, 'w') as f:
            f.write(result['classification_report'])
        
        cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.npy')
        np.save(cm_path, result['confusion_matrix'])
        
        print(f"Results saved for {model_name}")
    
    print(f"\nAll results saved to: {output_dir}")


def get_best_model(results_dict, metric='accuracy'):
    
    best_name = None
    best_score = -1
    
    for model_name, result in results_dict.items():
        metrics = result['metrics'] if 'metrics' in result else result
        score = metrics.get(metric, -1)
        
        if score > best_score:
            best_score = score
            best_name = model_name
    
    return best_name, best_score


def calculate_error_analysis(y_test, y_pred, y_pred_proba, threshold=0.5):
    
    errors = y_test != y_pred
    error_indices = np.where(errors)[0]
    
    false_positives = np.where((y_test == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]
    
    error_confidences = np.abs(y_pred_proba[errors] - threshold)
    
    analysis = {
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(y_test),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'avg_error_confidence': np.mean(error_confidences) if len(error_confidences) > 0 else 0,
        'error_indices': error_indices.tolist()
    }
    
    return analysis


def print_error_analysis(error_analysis):

    print("ERROR ANALYSIS")
    print(f"Total Errors:        {error_analysis['total_errors']}")
    print(f"Error Rate:          {error_analysis['error_rate']:.2%}")
    print(f"False Positives:     {error_analysis['false_positives']}")
    print(f"False Negatives:     {error_analysis['false_negatives']}")
    print(f"Avg Error Confidence: {error_analysis['avg_error_confidence']:.4f}")
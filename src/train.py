import os
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class KerasModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper para modelos Keras compatible con sklearn GridSearchCV.
    """
    
    def __init__(self, model_class=None, vocab_size=10000, embedding_dim=128, 
                 max_length=100, epochs=20, batch_size=32, verbose=0, **model_params):
        self.model_class = model_class
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_params = model_params
        self.model_ = None
        self.history_ = None
        
    def fit(self, X, y, validation_data=None):
        """Entrena el modelo"""
        if self.model_class is None:
            raise ValueError("model_class must be provided")
            
        model_obj = self.model_class(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            **self.model_params
        )
        
        model_obj.build()
        model_obj.compile()
        
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        if validation_data is not None:
            self.history_ = model_obj.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=validation_data,
                callbacks=callbacks_list,
                verbose=self.verbose
            )
        else:
            self.history_ = model_obj.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=callbacks_list,
                verbose=self.verbose
            )
        
        self.model_ = model_obj.model
        return self
    
    def predict(self, X):
        """Predice clases"""
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        y_pred_proba = self.model_.predict(X, verbose=0)
        return (y_pred_proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Predice probabilidades"""
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        y_pred = self.model_.predict(X, verbose=0).flatten()
        return np.column_stack([1 - y_pred, y_pred])
    
    def score(self, X, y):
        """Calcula accuracy para GridSearchCV"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self, deep=True):
        """Obtiene parámetros para GridSearchCV"""
        params = {
            'model_class': self.model_class,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }
        params.update(self.model_params)
        return params
    
    def set_params(self, **params):
        """Establece parámetros para GridSearchCV"""
        model_param_keys = ['dense_units', 'rnn_units', 'lstm_units', 
                           'dropout_rate', 'recurrent_dropout', 'l2_reg', 
                           'bidirectional']
        
        for key, value in params.items():
            if key in model_param_keys:
                self.model_params[key] = value
            else:
                setattr(self, key, value)
        return self


def tune_hyperparameters(model_class, X_train, y_train, param_grid,
                         vocab_size, max_length, embedding_dim=128,
                         epochs=20, batch_size=32, cv=3, verbose=1,
                         n_jobs=1, scoring='accuracy'):
    
    print(f"GRIDSEARCHCV - {model_class.__name__}")
    
    wrapper = KerasModelWrapper(
        model_class=model_class,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    grid_search = GridSearchCV(
        estimator=wrapper,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        error_score='raise'
    )
    
    # Calcular número de combinaciones manualmente
    n_combinations = 1
    for param_values in param_grid.values():
        n_combinations *= len(param_values)
    
    print(f"\nSearching over {n_combinations} parameter combinations...")
    print(f"Using {cv}-fold cross-validation")
    print(f"Scoring metric: {scoring}\n")
    
    start_time = datetime.now()
    grid_search.fit(X_train, y_train)
    search_time = (datetime.now() - start_time).total_seconds()
    
    print(f"GRIDSEARCH COMPLETED in {search_time:.2f} seconds")
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best {scoring}: {grid_search.best_score_:.4f}")
    
    results_df = _format_grid_results(grid_search.cv_results_)
    print(f"\n Top 5 Configurations:")
    print(results_df.head().to_string(index=False))
    
    return grid_search.best_params_, grid_search.best_score_, grid_search.cv_results_


def _format_grid_results(cv_results):
    """Formatea los resultados del GridSearch en un DataFrame"""
    import pandas as pd
    
    results = []
    for i in range(len(cv_results['mean_test_score'])):
        result = {
            'rank': cv_results['rank_test_score'][i],
            'mean_score': cv_results['mean_test_score'][i],
            'std_score': cv_results['std_test_score'][i],
            'params': str(cv_results['params'][i])
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('rank')
    return df


def train_and_evaluate(model_obj, model_name, X_train, y_train, X_test, y_test,
                       epochs=50, batch_size=32, validation_split=0.2,
                       callbacks_list=None, verbose=1):
    """
    Entrena y evalúa un modelo con los mejores hiperparámetros encontrados.
    """
    print(f"Training {model_name}")
    
    model_obj.build()
    model_obj.compile()
    
    print("\nModel Architecture:")
    model_obj.summary()
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = datetime.now()
    
    history = model_obj.model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks_list if callbacks_list else [],
        verbose=verbose
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n Training completed in {training_time:.2f} seconds")
    
    print("\nEvaluating on test set...")
    test_results = model_obj.model.evaluate(X_test, y_test, verbose=0)
    
    metrics_names = model_obj.model.metrics_names
    test_metrics = dict(zip(metrics_names, test_results))
    
    print("\n Test Set Results:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    save_training_history(history, model_name)
    
    return {
        'model': model_obj,
        'history': history,
        'test_metrics': test_metrics,
        'training_time': training_time
    }


def train_all_models(models_dict, X_train, y_train, X_test, y_test,
                    epochs=50, batch_size=32, validation_split=0.2,
                    patience=5, save_dir='outputs/saved_models'):
    """
    Entrena todos los modelos con sus mejores hiperparámetros.
    """
    from models import create_callbacks
    
    results = {}
    
    print("TRAINING ALL MODELS")
    
    for model_name, model_obj in models_dict.items():
        result = train_and_evaluate(
            model_obj, model_name,
            X_train, y_train, X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks_list=create_callbacks(model_name, patience=patience, 
                                           output_dir=save_dir),
            verbose=1
        )
        
        results[model_name] = result
    
    print(" ALL MODELS TRAINED SUCCESSFULLY")
    
    return results


def save_training_history(history, model_name, output_dir='../outputs/metrics'):
    """Guarda el historial de entrenamiento"""
    os.makedirs(output_dir, exist_ok=True)
    
    history_dict = {
        'history': history.history,
        'params': history.params,
        'epochs': len(history.history['loss'])
    }
    
    pickle_path = os.path.join(output_dir, f'{model_name}_history.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(history_dict, f)
    
    json_path = os.path.join(output_dir, f'{model_name}_history.json')
    json_dict = {
        'history': {k: [float(x) for x in v] for k, v in history.history.items()},
        'epochs': len(history.history['loss'])
    }
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"\n History saved to:")
    print(f"  - {pickle_path}")
    print(f"  - {json_path}")


def load_training_history(model_name, output_dir='../outputs/metrics'):
    """Carga el historial de entrenamiento"""
    pickle_path = os.path.join(output_dir, f'{model_name}_history.pkl')
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"History not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        history_dict = pickle.load(f)
    
    return history_dict


def get_best_epoch(history, metric='val_accuracy', mode='max'):
    """Obtiene la mejor época según una métrica"""
    if isinstance(history, dict):
        values = history['history'][metric]
    else:
        values = history.history[metric]
    
    if mode == 'max':
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    
    return best_idx + 1, values[best_idx]


def print_training_summary(results_dict):
    """Imprime resumen del entrenamiento"""
    print("TRAINING SUMMARY")
    
    for model_name, results in results_dict.items():
        print(f"\n{model_name}:")
        print(f"  Training Time: {results['training_time']:.2f}s")
        
        best_epoch, best_val_acc = get_best_epoch(
            results['history'], 
            metric='val_accuracy'
        )
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        
        print(f"  Test Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"    {metric}: {value:.4f}")
import os
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks
l2 = tf.keras.regularizers.l2


class DenseNN:
    def __init__(self, vocab_size, embedding_dim=128, max_length=100,
                 dense_units=[128, 64], dropout_rate=0.5, l2_reg=0.001):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None

    def build(self):
        self.model = models.Sequential([
            layers.Input(shape=(self.max_length,)),
            layers.Embedding(self.vocab_size, self.embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.dense_units[0], activation='relu', 
                        kernel_regularizer=l2(self.l2_reg)),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.dense_units[1], activation='relu', 
                        kernel_regularizer=l2(self.l2_reg)),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation='sigmoid')
        ])
        return self.model

    def compile(self, learning_rate=0.001):
        if self.model is None:
            self.build()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        return self.model

    def summary(self):
        if self.model is None:
            self.build()
        return self.model.summary()


class VanillaRNN:
    def __init__(self, vocab_size, embedding_dim=128, max_length=100,
                 rnn_units=64, dropout_rate=0.5, l2_reg=0.001):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None

    def build(self):
        self.model = models.Sequential([
            layers.Input(shape=(self.max_length,)),
            layers.Embedding(self.vocab_size, self.embedding_dim),
            layers.SimpleRNN(
                self.rnn_units,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg)
            ),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation='sigmoid')
        ])
        return self.model

    def compile(self, learning_rate=0.001):
        if self.model is None:
            self.build()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        return self.model

    def summary(self):
        if self.model is None:
            self.build()
        return self.model.summary()


class LSTMNetwork:
    def __init__(self, vocab_size, embedding_dim=128, max_length=100,
                 lstm_units=64, dropout_rate=0.5, recurrent_dropout=0.0,
                 l2_reg=0.001, bidirectional=False):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.bidirectional = bidirectional
        self.model = None

    def build(self):
        lstm_layer = layers.LSTM(
            self.lstm_units,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            kernel_regularizer=l2(self.l2_reg),
            recurrent_regularizer=l2(self.l2_reg)
        )

        layer_list = [
            layers.Input(shape=(self.max_length,)),
            layers.Embedding(self.vocab_size, self.embedding_dim),
        ]
        
        if self.bidirectional:
            layer_list.append(layers.Bidirectional(lstm_layer))
        else:
            layer_list.append(lstm_layer)
        
        layer_list.extend([
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model = models.Sequential(layer_list)
        return self.model

    def compile(self, learning_rate=0.001):
        if self.model is None:
            self.build()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        return self.model

    def summary(self):
        if self.model is None:
            self.build()
        return self.model.summary()


def create_callbacks(model_name, patience=5, monitor='val_loss', 
                    output_dir='outputs/saved_models'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f'{model_name}_best.keras')
    
    callback_list = [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callback_list


def get_model(model_type, vocab_size, embedding_dim=128, max_length=100, **kwargs):
    models_dict = {
        'dense': DenseNN,
        'rnn': VanillaRNN,
        'lstm': LSTMNetwork
    }

    if model_type not in models_dict:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(models_dict.keys())}"
        )

    return models_dict[model_type](
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_length=max_length,
        **kwargs
    )


def load_saved_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model
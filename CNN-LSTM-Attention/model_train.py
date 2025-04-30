import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, GRU, Dense, Input, Attention, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

class Config:
    def __init__(self):
        self.sequence_length = 230
        self.batch_size = 64
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.num_classes = 2
        self.hidden_size = 128
        self.rnn_type = 'lstm'
        self.bidirectional = True
        self.use_attention = True
        self.l2_reg = 0.0005
        self.dropout_rate = 0.5
        self.cnn_filters = 64
        self.cnn_kernel_size = 3

    # Method to convert configuration into a dictionary format for saving/loading
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        config.__dict__.update(config_dict)
        return config


def load_and_preprocess_data(csv_file, config):
    combined_beats_array = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    labels = combined_beats_array[:, 0]
    ecg_values = combined_beats_array[:, 1:]
    scaler = StandardScaler()
    ecg_values_array = scaler.fit_transform(ecg_values)

    X_train, X_test, y_train, y_test = train_test_split(
        ecg_values_array, labels, test_size=0.2, random_state=42, stratify=labels
    )
    y_train = to_categorical(y_train, num_classes=config.num_classes)
    y_test = to_categorical(y_test, num_classes=config.num_classes)

    return X_train, X_test, y_train, y_test


class FullModel(Model):
    def __init__(self, config):
        super(FullModel, self).__init__()
        self.config = config

        self.conv1 = Conv1D(filters=config.cnn_filters, kernel_size=config.cnn_kernel_size, activation='relu', kernel_regularizer=l2(config.l2_reg))
        self.pool = MaxPooling1D(pool_size=2)
        self.batch_norm = BatchNormalization()
        self.dropout_cnn = Dropout(config.dropout_rate)

        if config.rnn_type == 'lstm':
            self.rnn_layer = LSTM(config.hidden_size, return_sequences=config.use_attention, kernel_regularizer=l2(config.l2_reg))
        elif config.rnn_type == 'gru':
            self.rnn_layer = GRU(config.hidden_size, return_sequences=config.use_attention, kernel_regularizer=l2(config.l2_reg))

        if config.use_attention:
            self.attention = Attention()
            self.flatten = Flatten()

        self.dense = Dense(config.num_classes, activation='softmax', kernel_regularizer=l2(config.l2_reg))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.batch_norm(x)
        x = self.dropout_cnn(x, training=training)
        x = self.rnn_layer(x)

        if hasattr(self, 'attention'):
            attn_output = self.attention([x, x])
            x = self.flatten(attn_output)

        return self.dense(x)

    def get_config(self):
        config = super(FullModel, self).get_config()
        config.update({'config': self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config):
        model_config = config['config']
        return cls(Config.from_dict(model_config))


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        self.lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        self.train_losses = []
        self.val_losses = []

    def fit(self, train_data, val_data):
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            callbacks=[self.early_stopping, self.lr_scheduler]
        )
        self.train_losses = history.history['loss']
        self.val_losses = history.history['val_loss']

    def plot_loss(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Test Loss')
        plt.show()


def main():
    config = Config()
    csv_file = 'C:/Users/DELL/OneDrive/Documents/combined_beats.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file, config)

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    model = FullModel(config)
    trainer = Trainer(model, config)
    trainer.fit((X_train, y_train), (X_test, y_test))

    trainer.plot_loss()

    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=config.batch_size)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    model.save('C:/Users/DELL/OneDrive/Documents/trained_model.keras', save_format='keras')


if __name__ == '__main__':
    main()

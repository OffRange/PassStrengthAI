import keras

from ai import BaseModel, CHARS
from keras.models import Sequential
from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense
from keras.regularizers import l2
from keras.utils import to_categorical
import numpy as np

from dataset import Dataset


def one_hot_encode(text, max_length):
    encoding = np.zeros((max_length, len(CHARS)), dtype=np.float32)
    for i, char in enumerate(text):
        if char in CHARS:
            index = CHARS.index(char)
            encoding[i][index] = 1
    return encoding


class Model(BaseModel):
    def __init__(self, input_shape, num_classes, model: keras.models.Model = None):
        super().__init__(input_shape, num_classes, model)
        if model is not None:
            return

        self.model = Sequential([
            Input(shape=input_shape),

            # Convolutional Layers
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(1),
            Dropout(0.25),

            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(1),
            Dropout(0.5),

            Conv1D(256, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.25),

            # LSTM Layers
            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.25),

            Bidirectional(LSTM(256, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.25),

            Bidirectional(LSTM(512)),
            BatchNormalization(),
            Dropout(0.25),

            # Dense Layers
            Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.25),

            Dense(512, activation='tanh', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.25),

            Dense(256, activation='tanh', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.25),

            # Output layer
            Dense(num_classes, activation='softmax')
        ])

    def prepare_data(self, dataset: Dataset, max_length=30, num_classes=5):
        train_passwords_encoded = []
        for password in dataset.train_dataset[0]:
            train_passwords_encoded.append(one_hot_encode(str(password), max_length))

        train_labels = dataset.train_dataset[1]
        train_labels = to_categorical(train_labels, num_classes=num_classes) if train_labels is not None else []

        test_passwords_encoded = []
        test_labels = []
        if dataset.test_dataset:
            for password in dataset.test_dataset[0]:
                test_passwords_encoded.append(one_hot_encode(str(password), max_length))

            test_labels = dataset.test_dataset[1]
            test_labels = to_categorical(test_labels, num_classes=num_classes) if test_labels is not None else []

        return np.array(train_passwords_encoded), train_labels, np.array(test_passwords_encoded), test_labels
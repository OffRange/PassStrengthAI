import logging
import string
from datetime import datetime

import keras
import numpy as np
import tensorflow
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from tensorflow.lite.python.lite import TFLiteConverterV2, OpsSet

from dataset import Dataset, LABEL_LOOKUP

CHARS = string.ascii_letters + string.digits + string.punctuation


class BaseModel:
    def __init__(self, input_shape, num_classes, model: keras.models.Model = None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model: keras.models.Model = model

    def evaluate_model(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f'Test loss: {loss}, Test accuracy: {accuracy}')
        return loss, accuracy

    def predict_classes(self, x_prediction):
        y_prediction = self.model.predict(x_prediction)
        return [np.argmax(prediction) for prediction in y_prediction], [np.max(prediction) for prediction in
                                                                        y_prediction]

    def prepare_data(self, dataset: Dataset, max_length=30, num_classes=5):
        return list

    def save_model(self, folder='models/', filename="model.tflite"):
        path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f'Saving model to {folder} as {path}')
        self.model.save(f'{folder}/{path}')

        converter = TFLiteConverterV2.from_keras_model(self.model)
        converter.target_spec.supported_ops = [
            OpsSet.TFLITE_BUILTINS,
            OpsSet.SELECT_TF_OPS
        ]

        tflite_model = converter.convert()
        with open(f'{folder}/{path}/{filename}', 'wb') as f:
            f.write(tflite_model)

    @classmethod
    def load_from_file(cls, filepath: str):
        model: keras.models.Model = keras.models.load_model(filepath)
        return cls((model.input_shape[1], model.input_shape[2]), model.output_shape[1], model)


def try_model(model: BaseModel):
    try:
        while True:
            password = input('Enter a password: ')
            if password == '!exit':
                exit()

            if len(password) > model.input_shape[0]:
                print(f"The maximum input length that the model can handle is {model.input_shape[0]} characters")
                continue

            classes, percentage = model.predict_classes(
                model.prepare_data(Dataset(([password.strip()], []), ()), model.input_shape[0],
                                   num_classes=len(LABEL_LOOKUP))[0])
            print(classes, percentage[0] * 100, [LABEL_LOOKUP[predicted_class] for predicted_class in classes])
    except KeyboardInterrupt:
        pass


def train_model(model_class: type, dataset: Dataset, max_length: int):
    model: BaseModel = model_class((max_length, len(CHARS)), len(LABEL_LOOKUP))
    model.model.compile(Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    train_x, train_y, test_x, test_y = model.prepare_data(dataset, max_length=max_length, num_classes=len(LABEL_LOOKUP))

    model.model.fit(train_x, train_y, validation_data=(test_x, test_y), callbacks=[early_stopping], batch_size=128 * 4,
                    epochs=1)
    loss, acc = model.evaluate_model(test_x, test_y)
    return model, loss, acc

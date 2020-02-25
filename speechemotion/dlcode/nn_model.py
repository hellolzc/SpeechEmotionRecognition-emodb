import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import LSTM as KERAS_LSTM
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

from keras.utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler

from keras import backend as K

from speechemotion.mlcode.model_base_class import Model

def shuffle_train_data(X_train, Y_train):
    # 只打乱训练集
    shuffle_index = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle_index]
    Y_train = Y_train[shuffle_index]
    return X_train, Y_train, shuffle_index

def generate_arrays_from_data(X_train, Y_train, sample_size):
    # e.g.
    # X shape: (13368, 1000, 130) (1547, 1000, 130)
    # Y shape: (13368,) (1547,)
    steps_per_epoch = int(np.ceil(X_train.shape[0]/sample_size))
    while True:
        # 每个epoch做一次shuffle
        X_train, Y_train, _ = shuffle_train_data(X_train, Y_train)
        for j in range(steps_per_epoch):  # [0,1,...,steps_per_epoch-1]
            start_indx = j*sample_size
            end_indx = (j+1)*sample_size
            if end_indx > X_train.shape[0]:
                end_indx = X_train.shape[0]
            X_j = X_train[start_indx:end_indx, :]
            Y_j = Y_train[start_indx:end_indx]
            yield (X_j, Y_j)


class KerasModelAdapter(Model):
    """将keras模型装饰一下，从而将所有的模型设置集中到一处"""
    def __init__(self, input_shape=None, model_creator=None, **params):
        if 'name' not in params:
            params['name'] = 'KerasModelAdapter'
        super().__init__(params)
        self.params = params
        self.input_shape = input_shape
        self.model_creator = model_creator
        model = model_creator(input_shape)
        self.model = model
        self.train_history = None

    def set_hyper_params(self, lr=0.0001, loss='categorical_crossentropy'):
        pass


    def __str__(self):
        return self.summary()

    def _load_model(self, to_load):
        """
        Load the model weights from the given path.

        Args:
            to_load (str): path to the saved model file in h5 format.

        """
        try:
            self.model.load_weights(to_load)
        except:
            raise Exception("Invalid saved file provided")


    def save_model(self, save_path):
        """
        Save the model weights to `save_path` provided.
        """
        self.model.save_weights(save_path)

    def summary(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x), line_length=90)
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def plot_model(self, file_path='./log/model2.png'):
        plot_model(self.model, to_file=file_path, show_shapes=True)

    def _compile_model(self):
        """fit之前需要先compile"""
        opt = keras.optimizers.Adam(lr=0.0005)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, Y_train, validation_data=None, batch_size=32):
        """return None"""
        # history = _model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(val_x, val_y))
        # return self.model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=validation_data)
        # 
        Y_train = to_categorical(Y_train)
        if validation_data is not None:
            val_x, val_y = validation_data
            val_y = to_categorical(val_y)
            validation_data = (val_x, val_y)
        my_generator = generate_arrays_from_data(X_train, Y_train, batch_size)
        steps_per_epoch = int(np.ceil(X_train.shape[0]/batch_size))
        print("Shape:", X_train.shape[0], steps_per_epoch, batch_size)

        self._compile_model()
        self.train_history = self.model.fit_generator(my_generator, epochs=150,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=validation_data)
        self.show_history()

    def predict(self, X):
        return np.argmax(self.model.predict(X).squeeze(), axis=1)
        # return np.round(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict(X)

    def clone_model(self):
        """reset graph and return a deep copy of this model object"""
        # 清理内存，才能载入新模型，意味着旧模型报废了
        K.clear_session()
        new_model = KerasModelAdapter(input_shape=self.input_shape, model_creator=self.model_creator, **self.params)
        return new_model

    def show_history(self):
        """将训练过程可视化的函数"""
        history = self.train_history
        print(history.history.keys())
        fig = plt.figure(figsize=(15,4))

        ax = plt.subplot(121)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        ax.set_ylim([0.1, 1.0])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        ax = plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        # ax.set_ylim([0.0, 0.9])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower left')
        plt.show()

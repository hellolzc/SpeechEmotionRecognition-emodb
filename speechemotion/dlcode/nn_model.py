import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler

from keras.utils import plot_model
from keras import backend as K
from keras.layers import LSTM


def generate_arrays_from_data(X_train, Y_train, sample_size):
    # e.g.
    # X shape: (13368, 1000, 130) (1547, 1000, 130)
    # Y shape: (13368,) (1547,)
    steps_per_epoch = int(np.ceil(X_train.shape[0]/sample_size))
    while True:
        for j in range(steps_per_epoch):  # [0,1,...,steps_per_epoch-1]
            start_indx = j*sample_size
            end_indx = (j+1)*sample_size
            if end_indx > X_train.shape[0]:
                end_indx = X_train.shape[0]
            X_j = X_train[start_indx:end_indx, :]
            Y_j = Y_train[start_indx:end_indx]
            yield (X_j, Y_j)



def model_factory(input_shape, model_choose='cnn'):
    if model_choose == 'cnn_0':
        model = Sequential()
        # default "image_data_format": "channels_last",  input_shape = train_x.shape[1:]
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=[*input_shape, 1]))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(64, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))
        model.add(Convolution2D(64, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))
        model.add(Convolution2D(128, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

    elif model_choose == 'cnn_1':
        model = Sequential()
        # default "image_data_format": "channels_last"

        model.add(Convolution1D(128, 3, strides=2, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        for filter_num in [128, 128, 128]:
            model.add(Convolution1D(filter_num, 3, strides=2, padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

    elif model_choose == 'cnn':
        model = Sequential()
        # default "image_data_format": "channels_last"

        model.add(Convolution1D(64, 3, strides=1, input_shape=input_shape, padding='same',
                                use_bias=False, kernel_initializer='random_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        for filter_num in [64, 64, 64, 64, 64]:
            model.add(Convolution1D(filter_num, 3, strides=1, padding='same',
                                    use_bias=False, kernel_initializer='random_uniform'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(256, use_bias=False, kernel_initializer='random_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid'))

    elif model_choose == 'lstm':
        model = Sequential()
        model.add(Embedding(1547, 259))
        model.add(LSTM(400, dropout=0.10, return_sequences=True))
        model.add(Dense(256,activation='softmax', kernel_regularizer=regularizers.l2(0.001),
                            activity_regularizer=regularizer.l1(0.001)))
        modellstm.add(Dense(8, activation='tanh'))
        modellstm.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    else:
        raise Exception('NotImplementError')
    return model


class NN_MODEL():
    """将keras模型装饰一下，从而将所有的模型设置集中到一处"""
    def __init__(self, input_shape=None, model_creator=None):
        K.clear_session()
        self.input_shape = input_shape
        self.model_creator = model_creator
        model = model_creator(input_shape)
        self._model = model
        self.train_history = None

    def _compile_model(self):
        opt = keras.optimizers.Adam(lr=0.0001)
        self._model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    @property
    def model(self):
        print('[WARN] You should not use this model directly!')
        return self._model

    def __str__(self):
        return self.summary()

    def summary(self):
        stringlist = []
        self._model.summary(print_fn=lambda x: stringlist.append(x), line_length=90)
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def plot_model(self, file_path='./log/model2.png'):
        plot_model(self._model, to_file=file_path, show_shapes=True)

    @staticmethod
    def shuffle_train_data(X_train, Y_train):
        # 只打乱训练集
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        Y_train = Y_train[shuffle_index]
        return X_train, Y_train, shuffle_index

    def fit(self, X_train, Y_train, validation_data=None, batch_size=32):
        """return history"""
        # history = _model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(val_x, val_y))
        # return self._model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=validation_data)
        # 
        X_train, Y_train, _ = self.shuffle_train_data(X_train, Y_train)
        my_generator = generate_arrays_from_data(X_train, Y_train, batch_size)
        steps_per_epoch = int(np.ceil(X_train.shape[0]/batch_size))
        print("Shape:", X_train.shape[0], steps_per_epoch, batch_size)

        self._compile_model()
        self.train_history = self._model.fit_generator(my_generator, epochs=15,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=validation_data)
        return self

    def predict(self, X):
        return np.argmax(self._model.predict(X).squeeze())
        # return np.round(self._model.predict(X))

    def predict_proba(self, X):
        return self._model.predict(X)

    def clone(self):
        """reset graph and return a deep copy of this model object"""
        new_model = NN_MODEL(input_shape=self.input_shape, model_creator=self.model_creator)
        return new_model

    def show_history(self):
        """将训练过程可视化的函数"""
        history = self.train_history
        print(history.history.keys())
        fig = plt.figure(figsize=(15,4))

        ax = plt.subplot(121)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        ax.set_ylim([0.4, 0.9])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        ax = plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        ax.set_ylim([0.4, 0.9])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower left')
        plt.show()

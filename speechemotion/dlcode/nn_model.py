import numpy as np
import matplotlib.pyplot as plt
import inspect, html
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import LSTM as KERAS_LSTM
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

from keras.utils import to_categorical
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler, EarlyStopping

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
        """model_creator是一个函数，输入为input_shape, 返回一个Keras Model"""
        self.params = params.copy()  # 存档
        self.lr = params.pop('lr', 0.001)
        self.lr_decay = params.pop('lr_decay', 0.0)
        self.epochs = params.pop('epochs', 300)
        self.verbose = params.pop('verbose', 0)
        self.gpus = params.pop('gpus', 1)
        self.batch_size = params.pop('batch_size', 64 * self.gpus)
        self.loss = params.pop('loss', 'categorical_crossentropy')

        if 'name' not in params:
            params['name'] = 'KerasModelAdapter'
        super().__init__(params)  # 用不上的参数传给父类

        K.clear_session()  # 清理掉旧模型
        self.input_shape = input_shape
        self.model_creator = model_creator
        model = model_creator(input_shape)
        self.model = model
        self.train_history = None

    def set_hyper_params(self, **hyper_params):
        # TODO: finish this method
        raise NotImplementedError


    def __str__(self):
        return self.summary()

    def __repr__(self):
        try:
            func_src = inspect.getsource(self.model_creator)
            func_src = html.unescape(func_src)
        except IOError:
            func_src = "nocode"
        return repr({'code':func_src, 'params':self.params, 'input_shape':self.input_shape})

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
        stringlist.append('Params:' + str(self.params))
        self.model.summary(print_fn=lambda x: stringlist.append(x), line_length=90)
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def plot_model(self, file_path='./log/model2.png', show_shapes=True):
        plot_model(self.model, to_file=file_path, show_shapes=show_shapes)

    def _compile_model(self):
        """fit之前需要先compile"""
        if self.gpus > 1:
            self.model = multi_gpu_model(self.model, self.gpus)
        opt = keras.optimizers.Adam(lr=self.lr, decay=self.lr_decay)
        self.model.compile(optimizer=opt, loss=self.loss, metrics=['accuracy'])

    def fit(self, X_train, Y_train, validation_data=None):
        """return None"""
        # history = _model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(val_x, val_y))
        # return self.model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=validation_data)
        # 
        Y_train = to_categorical(Y_train)
        if validation_data is not None:
            val_x, val_y = validation_data
            val_y = to_categorical(val_y)
            validation_data = (val_x, val_y)
        batch_size = self.batch_size
        my_generator = generate_arrays_from_data(X_train, Y_train, batch_size)
        steps_per_epoch = int(np.ceil(X_train.shape[0]/batch_size))
        print("[INFO @ %s]"%__name__, "SampleNum:", X_train.shape[0], 'StepsPerEpoch:', steps_per_epoch,
            'Batchsize:', batch_size)
        if self.lr_decay != 0.0:
            new_lr = self.lr * 1 / (1 + self.lr_decay*steps_per_epoch)
            print("LearningRate will be %f after 1 epoch." % new_lr)
            new_lr = self.lr * 1 / (1 + self.lr_decay*steps_per_epoch*self.epochs)
            print("LearningRate will be %f after last epoch." % new_lr)

        self._compile_model()
        # es = EarlyStopping(monitor='val_loss', patience=5)
        self.train_history = self.model.fit_generator(my_generator,
                                    epochs=self.epochs, verbose=self.verbose,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=validation_data)  # callbacks=[es]
        self.trained = True
        self.show_history()

    def predict(self, X):
        return np.argmax(self.predict_proba(X).squeeze(), axis=1)
        # return np.round(self.model.predict(X)) # sigmoid

    def predict_proba(self, X):
        # avoid error "CUDNN_STATUS_BAD_PARAM" when using 2 gpu
        ori_length = len(X)
        padded_shape = list(X.shape)
        padded_shape[0] = int(np.ceil(ori_length / float(self.batch_size)) * self.batch_size)
        X_padded = np.zeros(padded_shape, dtype=X.dtype)
        X_padded[0:ori_length] = X
        return self.model.predict(X_padded)[0:ori_length]

    def clone_model(self):
        """reset graph and return a deep copy of this model object"""
        # 载入新模型，会自动清理内存，意味着旧模型报废了
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
        ax.set_ylim([0.2, 1.0])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        ax = plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        ax.set_ylim([0.0, 3.0])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower left')
        plt.show()

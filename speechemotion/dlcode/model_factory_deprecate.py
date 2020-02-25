
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import LSTM as KERAS_LSTM
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

def model_factory(input_shape, model_choose='cnn'):
    if model_choose == 'cnn_0':
        model = Sequential()
        # default "image_data_format": "channels_last",  input_shape = train_x.shape[1:]
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=[*input_shape, 1]))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

    elif model_choose == 'cnn_1':
        model = Sequential()
        # default "image_data_format": "channels_last"

        model.add(Conv1D(128, 3, strides=2, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        for filter_num in [128, 128, 128]:
            model.add(Conv1D(filter_num, 3, strides=2, padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

    elif model_choose == 'cnn':
        model = Sequential()
        # default "image_data_format": "channels_last"

        model.add(Conv1D(64, 3, strides=1, input_shape=input_shape, padding='same',
                                use_bias=False, kernel_initializer='random_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        for filter_num in [64, 64, 64, 64, 64]:
            model.add(Conv1D(filter_num, 3, strides=1, padding='same',
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
    else:
        raise Exception('NotImplementError')
    return model
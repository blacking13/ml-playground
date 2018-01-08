import numpy as np
from keras.models import Model
import keras.layers
from keras.utils import np_utils
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
import os
import pickle
import traceback


def load_file(filename):

    num_channels = 3
    img_size = 32

    # Create full path for the file.
    file_path = filename

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    raw_images = data[b'data']

    raw_images = np.array(raw_images, dtype=float) / 255.0
    raw_images = raw_images.reshape([-1, num_channels, img_size, img_size])
    raw_images = raw_images.transpose([0, 2, 3, 1])

    cls = np.array(data[b'labels'])

    return raw_images, cls


def train(X_train, y_train, X_test, y_test):
    batch_size = 32  # 32
    num_epoch = 20  # 200
    kernel_size = 3
    pool_size = 2

    conv_depth_1 = 16
    conv_depth_2 = 64
    drop_prob_1 = 0.25
    drop_prob_2 = 0.5
    hidden_size = 512

    num_train, height, width, depth = X_train.shape
    num_test = X_train.shape[0]
    num_classes = np.unique(y_train).shape[0]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= np.max(X_train)
    X_test /= np.max(X_test)

    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)

    input = Input(shape=(height, width, depth))
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(input)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)

    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)

    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size)(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    output = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=2, validation_split=0.1)
    res = model.evaluate(X_test, Y_test, verbose=1)

    print("Evaluation: {}".format(res))
    print(model.summary())
    return model

try:
    prefix = '/opt/ml/'
    channel_name = 'training';

    input_path = os.path.join(prefix, 'input/data', channel_name)
    output_path = os.path.join(prefix, 'output')
    model_path = os.path.join(prefix, 'model')
    param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

    X_batches = []
    Y_batches = []

    for i in range(5):
        file_name = os.path.join(input_path, 'data_batch_' + str(i + 1))
        X_train_t, y_train_t = load_file(file_name)
        X_batches.append(X_train_t)
        Y_batches.append(y_train_t)

    X_train = np.concatenate(X_batches)
    y_train = np.concatenate(Y_batches)
    print('X_train.shape: {}; y_train.shape: {};'.format(X_train.shape, y_train.shape))

    file_name = os.path.join(input_path, 'test_batch')
    X_test, y_test = load_file(file_name)
    print('finished loading data')

    model = train(X_train, y_train, X_test, y_test)

    output_model_file = os.path.join(model_path, 'output.h5')
    model.save(output_model_file)

    print('Finished training')
except Exception as e:
    trc = traceback.format_exc()
    print('Exception during training: ' + str(e) + '\n' + trc)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, MaxPooling2D, TimeDistributed, Bidirectional, LSTM, Conv2D, ZeroPadding2D, Permute, Reshape, GRU

K.set_image_data_format('channels_last')

LAST_CNN_SIZE=256

def KerasCrnn(img_height, img_width, n_labels, nc=1, nh=8):
    '''
    Convolutional recurrent neural network inspired by this paper https://arxiv.org/pdf/1602.05875.pdf
    and implemented based on Harish Karumuthil's code here https://github.com/harish2704/pottan-ocr
    
    Inputs:
        - img_height height of image in pixels
        - n_labels number of different labels to classify images into
        - nc no idea but it defaults to 1
        - nh size of LSTM hidden state
    '''

    ks = [3  , 3  , 3   , 3   , 3   , 3   ] #, 2  ]
    ps = [1  , 1  , 1   , 1   , 1   , 1   ] #, 0   ]
    ss = [1  , 1  , 1   , 1   , 1   , 1   ] #, 1   ]
    nm = [32 , 64 , 128 , 128 , 128 , 128 ] #, 512 ]

    model = Sequential()

    def convRelu(i, batchNormalization=False):
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        padding = 'same' if ps[i] else 'valid'

        if( i == 0):
            model.add(Conv2D(nOut, ks[i], strides=ss[i], input_shape=(img_width, img_height, 3), padding=padding, name='conv{0}'.format(i)))
            # model.add(Conv2D(nOut, ks[i], strides=ss[i], input_shape=(img_height, None, 1), padding=padding, name='conv{0}'.format(i)))
        else:
            model.add(Conv2D(nOut, ks[i], strides=ss[i], padding=padding, name='conv{0}'.format(i)))

        if batchNormalization:
            model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='batchnorm{0}'.format(i)))
        model.add(Activation('relu', name='relu{0}'.format(i)))

    convRelu(0)
    model.add( MaxPooling2D(pool_size=2, strides=2, name='pooling{0}'.format(0)))  # 64x16x64
    convRelu(1)
    model.add( MaxPooling2D(pool_size=2, strides=2, name='pooling{0}'.format(1)))  # 128x8x32
    convRelu(2, True)
    convRelu(3)
    model.add( ZeroPadding2D(padding=(0,1)))
    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pooling{0}'.format(2)))  # 256x4x16

    convRelu(4, True)
    convRelu(5)
    model.add( ZeroPadding2D(padding=(0,1)))
    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pooling{0}'.format(3)))  # 512x2x16

    model.add(Reshape((-1, LAST_CNN_SIZE)))
    model.add(Bidirectional(LSTM(nh, return_sequences=True, use_bias=True, recurrent_activation='sigmoid', )))
    model.add(TimeDistributed(Dense(nh)))
    model.add(Bidirectional(LSTM(nh, return_sequences=True, use_bias=True, recurrent_activation='sigmoid', )))
    # model.add(TimeDistributed(Dense(n_labels, activation='softmax')))
    model.add(Dense(n_labels, activation='softmax'))

    return model
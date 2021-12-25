import tensorflow as tf
from tensorflow.keras import Input, Conv2D, ReLU, MaxPool2D, Dropout, Dense, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Model

def AlexNet(args):
    input_layer = Input(shape=args.target_size)

    # 1st convolutional layer
    x = Conv2D(
        filters=96,
        kenel_size=(11, 11),
        input_shape=(224, 224, 3),
        strides=(4, 4),
        padding='valid'
    )(input_layer)
    x = Activation('relu')(x)
    x = MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(x)
    x = BatchNormalization(axis=3)(x)

    # 2nd convolutional layer
    x = Conv2D(
        filters=256,
        kenel_size=(11, 11),
        strides=(1, 1),
        padding='valid'
    )(x)
    x = Activation('relu')(x)
    x = MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(x)
    x = BatchNormalization(axis=3)(x)

    # 3rd convolutional layer
    x = Conv2D(
        filters=384,
        kenel_size=(3, 3),
        strides=(1, 1),
        padding='valid'
    )(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3)(x)

    # 4th convolutional layer
    x = Conv2D(
        filters=384,
        kenel_size=(3, 3),
        strides=(1, 1),
        padding='valid'
    )(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3)(x)

    # 5th convolutional layer
    x = Conv2D(
        filters=256,
        kenel_size=(3, 3),
        strides=(1, 1),
        padding='valid'
    )(x)
    x = Activation('relu')(x)
    x = MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(x)
    x = BatchNormalization(axis=3)(x)

    # Flatten layer
    x = Flatten()(x)

    # 1st Dense layer
    x = Dense(4096, input_shape=(224, 224, 3))(x)
    x = Activation('relu')(x)
    x = Dropout(rate=.4)(x)
    x = BatchNormalization()(x)

    # 2nd dense layer
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=.4)(x)
    x = BatchNormalization(x)

    # Output layer
    output = Dense(args.classes, activation='softmax')(x)

    return Model(input_layer, output)


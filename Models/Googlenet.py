import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, ZeroPadding2D, MaxPooling2D, concatenate, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.nn import local_response_normalization

def inception_block(input_layer, filters):
    path1 = Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    path2 = Conv2D(filters=filters[1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu')(path2)

    path3 = Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=filters[4], kernel_size=(5, 5), padding='same', activation='relu')(path3)

    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=filters[5], kernel_size=(1, 1), padding='same', activation='relu')(path4)

    output = concatenate([path1, path2, path3, path4], axis=-1)
    return output


def GoogleNet(args):

    input_layer = Input(shape=(args.target_size))

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = inception_block(x, [192, 96, 208, 16, 48, 64])

    # output layer 1
    output1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    output1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(output1)
    output1 = Flatten()(output1)
    output1 = Dense(1024, activation='relu')(output1)
    output1 = Dropout(.7)(output1)
    output1 = Dense(args.classes, activation='softmax')(output1)

    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])

    # output layer 2
    output2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    output2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(output2)
    output2 = Flatten()(output2)
    output2 = Dense(1024, activation='relu')(output2)
    output2 = Dropout(.7)(output2)
    output2 = Dense(args.classes, activation='softmax')(output2)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    output3 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    output3 = Flatten()(output3)
    output3 = Dense(1024, activation='relu')(output3)
    output3 = Dropout(.7)(output3)
    output3 = Dense(args.classes, activation='softmax')(output3)

    model = Model(input=input_layer, output=[output1, output2, output3])
    return model

    






import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling2D, Input, Conv2D, ZeroPadding2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.models import Model

def convolutional_block(x, filters, kernels):
    for index in range(len(filters)):
        x = Conv2D(filters[index], kernels[index], strides=(1, 1), activation='relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x

def output_block(x, args):
    output = Flatten()(x)
    output = Dense(1024, activation='relu')(output)
    output = Dropout(.5)(output)
    output = Dense(args.classes, activation='softmax')(output)
    return output

def VGG11(args):
    input_layer = Input(shape=args.target_size)
    x = convolutional_block(input_layer, [64], [3]) 
    x = convolutional_block(x, [128], [3]) 
    x = convolutional_block(x, [256, 256], [3, 3])
    x = convolutional_block(x, [512, 512], [3, 3])
    x = convolutional_block(x, [512, 512], [3, 3])
    output_layer = output_block(x, args)
    return Model(input_layer, output_layer)

def VGG13(args):
    input_layer = Input(shape=args.target_size)
    x = convolutional_block(input_layer, [64, 64], [3, 3])
    x = convolutional_block(x, [128, 128], [3, 3])
    x = convolutional_block(x, [256, 256], [3, 3])
    x = convolutional_block(x, [512, 512], [3, 3])
    x = convolutional_block(x, [512, 512], [3, 3])
    output_layer = output_block(x, args)
    return Model(input_layer, output_layer)

def VGG16_1(args):
    input_layer = Input(shape=args.target_size)
    x = convolutional_block(input_layer, [64, 64], [3, 3])
    x = convolutional_block(x, [128, 128], [3, 3])
    x = convolutional_block(x, [256, 256, 256], [3, 3, 1])
    x = convolutional_block(x, [512, 512, 512], [3, 3, 1])
    x = convolutional_block(x, [512, 512, 512], [3, 3, 1])
    output_layer = output_block(x, args)
    return Model(input_layer, output_layer)

def VGG16(args):
    input_layer = Input(shape=args.target_size)
    x = convolutional_block(input_layer, [64, 64], [3, 3])
    x = convolutional_block(x, [128, 128], [3, 3])
    x = convolutional_block(x, [256, 256, 256], [3, 3, 3])
    x = convolutional_block(x, [512, 512, 512], [3, 3, 3])
    x = convolutional_block(x, [512, 512, 512], [3, 3, 3])
    output_layer = output_block(x, args)
    return Model(input_layer, output_layer)

def VGG19(args):
    input_layer = Input(shape=args.target_size)
    x = convolutional_block(input_layer, [64, 64], [3, 3])
    x = convolutional_block(x, [128, 128], [3, 3])
    x = convolutional_block(x, [256, 256, 256, 256], [3, 3, 3, 3])
    x = convolutional_block(x, [512, 512, 512, 512], [3, 3, 3, 3])
    x = convolutional_block(x, [512, 512, 512, 512], [3, 3, 3, 3])
    output_layer = output_block(x, args)
    return Model(input_layer, output_layer)
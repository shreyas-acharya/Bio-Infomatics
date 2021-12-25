import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, BatchNormalization, add, Activation, Conv2D, Input, ZeroPadding2D, AveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from tensorflow.python.ops.gen_batch_ops import Batch

def identity_block(x_input, filters, kernel):
    x = Conv2D(filters[0], kernel[0])(x_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel[1], padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel[2])(x)
    x = BatchNormalization(axis=3)(x)
    x = add([x, x_input])
    x = Activation('relu')(x)
    return x

def convolution_block(x_input, filters, kernel, strides=(2, 2)):
    x = Conv2D(filters[0], kernel[0], strides=strides)(x_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel[1], padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel[2])(x)
    x = BatchNormalization(axis=3)(x)

    x_skip = Conv2D(filters[2], (1, 1), strides=strides)(x_input)
    x_skip = BatchNormalization(axis=3)(x_skip)

    x = add([x, x_skip])
    x = Activation('relu')(x)
    return x

def ResNet(args, pattern, count, kernel):
    input_layer = Input(shape=args.target_size)

    x = ZeroPadding2D(padding=(3, 3))(input_layer)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for index in range(len(pattern)):
        x = convolution_block(x, pattern[index], kernel, strides=(1, 1))
        for _ in range(count[index]-1):
            x = identity_block(x, pattern[index], kernel)

    x = GlobalAveragePooling2D()(x)
    output = Dense(args.classes, activation='softmax')(x)
    return Model(input_layer, output)

def ResNet50(args):
    model = ResNet(
        args, 
        pattern=[[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]], 
        count=[3, 4, 6, 3], 
        kernel=[1, 3, 1]
    )
    return model

def ResNet101(args):
    model = ResNet(
        args, 
        pattern=[[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]], 
        count=[3, 4, 23, 3], 
        kernel=[1, 3, 1]
    )
    return model

def ResNet152(args):
    model = ResNet(
        args, 
        pattern=[[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]], 
        count=[3, 8, 36, 3], 
        kernel=[1, 3, 1]
    )
    return model

def ResNet34(args):
    model = ResNet(
        args, 
        pattern=[[64, 64], [128, 128], [256, 256], [512, 512]], 
        count=[3, 4, 6, 3], 
        kernel=[3, 3]
    )
    return model

def ResNet18(args):
    model = ResNet(
        args, 
        pattern=[[64, 64], [128, 128], [256, 256], [512, 512]], 
        count=[2, 2, 2, 2], 
        kernel=[3, 3]
    )
    return model




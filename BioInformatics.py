import argparse
import Models
from Models.VGG import VGG16
from Utils import load_data
from Models.ResNet import ResNet50
from Models.MobileNet import MobileNet
from Models.AlexNet import AlexNet
from Models.Googlenet import GoogleNet
def tuple_argument(data):
    return tuple(data)

def argument():
    args = argparse.ArgumentParser(description="BioInformatics")
    args.add_argument('--dataset_path', type=str, default='./Dataset/', help='Path to the dataset')
    args.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    args.add_argument('--target_size', nargs=3, default=(224, 224), help='Size of the Input image')
    args.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    args.add_argument('--classes', type=int, default=9, help='Number of Classes')
    args = args.parse_args()
    return args

def ResModel(args):
    model = ResNet50(args)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    training_set, test_set=load_data(args)
    model.fit(
    training_set,
    validation_data=test_set,
    epochs=25)

def VGGModel(args):
    model = VGG16(args)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    training_set, test_set=load_data(args)
    model.fit(
    training_set,
    validation_data=test_set,
    epochs=25)

def MobileModel(args):
    n_classes = 9
    input_shape = (224,224,3)
    model = MobileNet(input_shape,n_classes)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    training_set, test_set=load_data(args)
    model.fit(
    training_set,
    validation_data=test_set,
    epochs=25)

def AlexModel(args):
    model = AlexNet(args)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    training_set, test_set=load_data(args)
    model.fit(
    training_set,
    validation_data=test_set,
    epochs=25)

def GoogleModel(args):
    model = GoogleNet(args)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    training_set, test_set=load_data(args)
    model.fit(
    training_set,
    validation_data=test_set,
    epochs=25)


def main():
    args = argument()
    ResModel(args)
    MobileModel(args)
    AlexModel(args)
    VGGModel(args)
    GoogleModel(args)
    

main()

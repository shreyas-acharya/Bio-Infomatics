import argparse
from ResNet import ResNet50

def tuple_argument(data):
    return tuple(data)

def argument():
    args = argparse.ArgumentParser(description="BioInformatics")
    args.add_argument('--dataset_path', type=str, default='./Dataset/', help='Path to the dataset')
    args.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    args.add_argument('--target_size', nargs=3, default=(224, 224, 3), help='Size of the Input image')
    args.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    args.add_argument('--classes', type=int, default=9, help='Number of Classes')
    args = args.parse_args()
    return args


def main():
    print('hello')
    args = argument()
    model = ResNet50(args)
    model.summary()

main()
from keras.preprocessing.image import ImageDataGenerator





def load_data(args):
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(f'{args.dataset_path}/Train',
                                                    target_size = args.target_size,
                                                    batch_size = args.batch_size,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(f'{args.dataset_path}/Test',
                                                target_size = args.target_size,
                                                batch_size = args.batch_size,
                                                class_mode = 'categorical')
    return training_set, test_set
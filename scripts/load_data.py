# load_data.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(model_type="cnn"):

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_dir = "tomato_dataset/train"
    val_dir = "tomato_dataset/val"
    test_dir = "tomato_dataset/test"

    # ======================
    # PREPROCESSING
    # ======================
    if model_type == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            #'preprocess_input()' bcoz MobileNet is trained on images processed in a specific way:
                # Pixel values scaled to [-1, 1]
                # Not just 0–1
    # Agumenting helps learn better, increase data diversity and reduce overfitting
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)   
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  

    else:  # CNN
        train_datagen = ImageDataGenerator(
            rescale=1./255,     # Converts pixel values from 0–255 → 0–1
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(rescale=1./255)        # Converts pixel values from 0–255 → 0–1
        test_datagen = ImageDataGenerator(rescale=1./255)       # and NOT augmented for fair evaluation and need real unchanged image

    # ======================
    # GENERATORS        # Reads images from folders, assigns labels, and outputs batches like (image, label)
    # ======================
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator



# Found 3061 images belonging to 3 classes.
# Found 900 images belonging to 3 classes.
# Found 539 images belonging to 3 classes.

# Class indices: {'Tomato___Early_blight': 0,
# 'Tomato___Late_blight': 1,
# 'Tomato___healthy': 2}

# Tomato___Early_blight 680
# Tomato___healthy 1083
# Tomato___Late_blight 1298

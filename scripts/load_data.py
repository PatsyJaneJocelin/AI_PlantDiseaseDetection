from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Main function to load train, validation and test datasets
def get_data_generators(model_type="cnn"):

    IMG_SIZE = (224, 224)    # Standard image size for all models
    BATCH_SIZE = 32    # Number of images processed at once

    # Dataset paths
    train_dir = "tomato_dataset/train"
    val_dir = "tomato_dataset/val"
    test_dir = "tomato_dataset/test"

    # ======================
    # PREPROCESSING
    # ======================

    # If using MobileNet model
    if model_type == "mobilenet":
        # MobileNet requires special preprocessing
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            #'preprocess_input()':
                # Converts pixel values to range [-1, 1]  (generally the pixel range is [0,1])
                # because MobileNet was trained this was

        # Training data with agumentation
        # Agumenting helps learn better, increase data diversity, improve generalization, and reduce overfitting
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,      # Random rotation
            zoom_range=0.2,     # Random zoom
            horizontal_flip=True    # Random horizontal flip        
        )

        # Validation & Test data (NO Agumentation)
        # # Only preprocessing for fair evaluation and need real unchanged image
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)   
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  

    # CNN Preprocessing
    else:  
        # Training data with agumentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,     # Converts pixel values from 0–255 → 0–1
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )

        # Validation & Test data (NO Agumentation)
        val_datagen = ImageDataGenerator(rescale=1./255)        
        test_datagen = ImageDataGenerator(rescale=1./255)       

    # ======================
    # DATA GENERATORS        # Reads images from folders, assigns labels, and outputs batches like (image, label)
    # ======================
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,       # Resize all images
        batch_size=BATCH_SIZE,
        class_mode='categorical'        # Multi-class classification
    )

    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False       # Keeps image order fixed
    )

    return train_generator, val_generator, test_generator


# ======================
# DATASET INFO
# ======================

# Found 3061 images belonging to 3 classes.
# Found 900 images belonging to 3 classes.
# Found 539 images belonging to 3 classes.

# Class indices: {'Tomato___Early_blight': 0,
# 'Tomato___Late_blight': 1,
# 'Tomato___healthy': 2}

# Tomato___Early_blight 680
# Tomato___healthy 1083
# Tomato___Late_blight 1298

import os
import shutil
import random 

# Original dataset path
base_dir = "dataset"

# New dataset path
new_base = "tomato_dataset"

# Classes we want to keep
classes = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

# Existing dataset splits
splits = ["train", "val"]

# ======================
# COPY SELECTED CLASSES
# ======================

for split in splits:
    for cls in classes:
        src = os.path.join(base_dir, split, cls)        # Source folder path
        dst = os.path.join(new_base, split, cls)        # Destination folder path

        if not os.path.exists(src):        # Skip if source folder doesn't exist
            print(f"Skipping missing folder: {src}")
            continue

        os.makedirs(dst, exist_ok=True)        # Create destination folder if needed

        for file in os.listdir(src):        # Copy only image files
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):        # Check valid image extension
                shutil.copy(os.path.join(src, file), dst)       # Copy image to new dataset

print("✅ Data preparation completed!")         

# ======================
# CREATE TEST DATA
# ======================

# Test dataset folder
test_dir = "tomato_dataset/test"

for cls in classes:
    train_cls_path = os.path.join("tomato_dataset/train", cls)    # Path of training images
    test_cls_path = os.path.join(test_dir, cls)    # Path where test images will go

    os.makedirs(test_cls_path, exist_ok=True)    # Create test folder if needed

    images = os.listdir(train_cls_path)    # Get all images from train folder
    random.shuffle(images)    # Shuffle images randomly

    split_size = int(0.15 * len(images))      # 15% of train images moved to test set

    test_images = images[:split_size]    # Select first 15% images

    # Move selected images to test folder
    for img in test_images:
        shutil.move(
            os.path.join(train_cls_path, img),
            os.path.join(test_cls_path, img)
        )

print("✅ Test dataset created!")
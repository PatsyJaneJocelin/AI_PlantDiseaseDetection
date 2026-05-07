import os
import shutil
import random 

# Original dataset path
base_dir = "dataset"

# New dataset path
new_base = "tomato_dataset"

classes = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

splits = ["train", "val"]

for split in splits:
    for cls in classes:
        src = os.path.join(base_dir, split, cls)
        dst = os.path.join(new_base, split, cls)

        if not os.path.exists(src):         # Avoid crashing if a folder doesn’t exist
            print(f"Skipping missing folder: {src}")
            continue

        os.makedirs(dst, exist_ok=True)

        for file in os.listdir(src):        # Avoid copying non-image files
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(src, file), dst)

print("✅ Data preparation completed!")


# Creating test data
test_dir = "tomato_dataset/test"

for cls in classes:
    train_cls_path = os.path.join("tomato_dataset/train", cls)
    test_cls_path = os.path.join(test_dir, cls)

    os.makedirs(test_cls_path, exist_ok=True)

    images = os.listdir(train_cls_path)
    random.shuffle(images)

    split_size = int(0.15 * len(images))  # 15% from training data goes to testing data

    test_images = images[:split_size]

    for img in test_images:
        shutil.move(
            os.path.join(train_cls_path, img),
            os.path.join(test_cls_path, img)
        )

print("✅ Test dataset created!")
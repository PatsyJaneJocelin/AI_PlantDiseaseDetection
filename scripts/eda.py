from load_data import get_data_generators

import matplotlib.pyplot as plt
import os
import numpy as np

from tensorflow.keras.preprocessing import image

# ======================
# CREATE RESULTS FOLDER
# ======================
os.makedirs("results", exist_ok=True)

# ======================
# LOAD DATA
# ======================
train_generator, val_generator, test_generator = get_data_generators()

# ======================
# CLASS MAPPING
# ======================
print("Class indices:")
print(train_generator.class_indices)

# ======================
# CLASS DISTRIBUTION
# ======================
train_dir = "tomato_dataset/train"

print("\nClass Distribution (Train):")

for cls in os.listdir(train_dir):
    path = os.path.join(train_dir, cls)
    print(cls, "→", len(os.listdir(path)))

# ======================
# SAMPLE AUGMENTED IMAGES
# ======================
images, labels = next(train_generator)

plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.axis('off')

plt.suptitle("Sample Augmented Images")

augmented_path = "results/sample_augmented_images.png"

plt.savefig(augmented_path)
plt.show()

print(f"\n✅ Augmented image samples saved at: {augmented_path}")

# ======================
# ORIGINAL vs AUGMENTED
# ======================

# Choose class
sample_class = "Tomato___Early_blight"

sample_dir = os.path.join(train_dir, sample_class)

# Take first 3 images
sample_images = os.listdir(sample_dir)[:3]

plt.figure(figsize=(10, 12))

for idx, img_name in enumerate(sample_images):

    img_path = os.path.join(sample_dir, img_name)

    # ----------------------
    # LOAD ORIGINAL IMAGE
    # ----------------------
    original_img = image.load_img(
        img_path,
        target_size=(224,224)
    )

    original_array = image.img_to_array(original_img)

    # Normalize for display
    original_display = original_array / 255.0

    # ----------------------
    # GENERATE AUGMENTED IMAGE
    # ----------------------
    aug_input = np.expand_dims(original_array, axis=0)

    augmented_img = next(
        train_generator.image_data_generator.flow(
            aug_input,
            batch_size=1
        )
    )[0]

    # ----------------------
    # PLOT ORIGINAL
    # ----------------------
    plt.subplot(len(sample_images), 2, idx*2 + 1)

    plt.imshow(original_display)

    plt.title("Original")

    plt.axis('off')

    # ----------------------
    # PLOT AUGMENTED
    # ----------------------
    plt.subplot(len(sample_images), 2, idx*2 + 2)

    plt.imshow(augmented_img)

    plt.title("Augmented")

    plt.axis('off')

plt.suptitle(
    "Original vs Augmented Images",
    fontsize=16
)

plt.tight_layout()

comparison_path = "results/original_vs_augmented.png"

plt.savefig(comparison_path)

plt.show()

print(f"\n✅ Original vs Augmented comparison saved at: {comparison_path}")
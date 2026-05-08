# Import data generators from load_data.py
from load_data import get_data_generators

# Libraries for plotting, file handling, arrays, and image processing
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# ======================
# CREATE RESULTS FOLDER
# ======================
# Creates 'results' folder if it doesn't already exist
os.makedirs("results", exist_ok=True)

# ======================
# LOAD DATA
# ======================
# Load train, validation, and test generators
train_generator, val_generator, test_generator = get_data_generators()

# ======================
# CLASS MAPPING
# ======================
# Print class labels and their assigned numeric indices
print("Class indices:")
print(train_generator.class_indices)

# ======================
# CLASS DISTRIBUTION
# ======================
# Path to training dataset
train_dir = "tomato_dataset/train"

print("\nClass Distribution (Train):")

# Count number of images in each class folder
for cls in os.listdir(train_dir):
    path = os.path.join(train_dir, cls)
    print(cls, "→", len(os.listdir(path)))

# ======================
# SAMPLE AUGMENTED IMAGES
# ======================
# Get one batch of augmented images from training generator
images, labels = next(train_generator)

# Create figure for visualization
plt.figure(figsize=(10,5))

# Display first 6 augmented images
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

# Choose a sample class for comparison
sample_class = "Tomato___Early_blight"

# Path to selected class folder
sample_dir = os.path.join(train_dir, sample_class)

# Take first 3 images from folder
sample_images = os.listdir(sample_dir)[:3]

plt.figure(figsize=(10, 12))

# Loop through selected images
for idx, img_name in enumerate(sample_images):

    img_path = os.path.join(sample_dir, img_name)       # Full image path

    # ----------------------
    # LOAD ORIGINAL IMAGE
    # ----------------------
    # Load image and resize to 224x224
    original_img = image.load_img(
        img_path,
        target_size=(224,224)
    )

    # Convert image to array
    original_array = image.img_to_array(original_img)

    # Normalize pixel for display
    original_display = original_array / 255.0

    # ----------------------
    # GENERATE AUGMENTED IMAGE
    # ----------------------
    # Expand dimensions because augmentation expects batch input
    aug_input = np.expand_dims(original_array, axis=0)

    # Generate one agumented image
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
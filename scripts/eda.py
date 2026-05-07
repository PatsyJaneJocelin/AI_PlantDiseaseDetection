# eda.py

from load_data import get_data_generators
import matplotlib.pyplot as plt
import os

# ======================
# LOAD DATA
# ======================
train_generator, val_generator, test_generator = get_data_generators()

# ======================
# CLASS MAPPING
# ======================
print("Class indices:", train_generator.class_indices)


# ======================
# VISUALIZE IMAGES
# ======================
images, labels = next(train_generator)

plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.axis('off')

plt.suptitle("Sample Augmented Images")
plt.show()


# ======================
# CLASS DISTRIBUTION
# ======================
train_dir = "tomato_dataset/train"

print("\nClass Distribution (Train):")

for cls in os.listdir(train_dir):
    path = os.path.join(train_dir, cls)
    print(cls, "→", len(os.listdir(path)))
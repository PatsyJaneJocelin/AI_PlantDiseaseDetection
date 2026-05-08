import tensorflow as tf
from tensorflow.keras import layers, models
from load_data import get_data_generators

import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Creates folders if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ======================
# LOAD DATA
# ======================
# Load train, validation, and test datasets
train_generator, val_generator, test_generator = get_data_generators(model_type="cnn")

# ======================
# BUILD MODEL
# ======================
def build_model():
    # Sequential model = layers added one after another
    model = models.Sequential([
        tf.keras.Input(shape=(224,224,3)),      # Input image size

        # Convolution Block 1
        # Detects simple features (edges, textures)
        layers.Conv2D(32, (3,3), activation='relu'),    # relu makes 0 and -ve numbers 0 and keeps positive numbers as is (helps keep only features that add value)
        layers.MaxPooling2D(2,2),       # Reduces image size

        # Convolution Block 2
        # Learns more complex features
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        # Convolution Block 3
        # Learns deeper image patterns
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),        # Converts 2D feature maps → 1D vector

        layers.Dense(128, activation='relu'),        # Fully connected layer
        layers.Dropout(0.5),        # Automatically drops 50% neurons to prevent overfitting

        # Output layer
        # 3 classes --> softmax probabilities
        layers.Dense(3, activation='softmax')
    ])
    return model

model = build_model()       # Create model

# ======================
# COMPILE MODEL
# ======================
model.compile(
    optimizer='adam',       # Adjusts weights during training to reduce error
    loss='categorical_crossentropy',        # Measures the error
    metrics=[                                   # Evaluates performance
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),        # Precision = correct positive predictions
        tf.keras.metrics.Recall(name='recall')        # Recall = correctly detected positives
    ]
)

# ======================
# CALLBACKS
# ======================
# Stops training if validation loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',    # Monitor validation loss
    patience=3,    # Wait 3 epochs before stopping
    restore_best_weights=True    # Restore best model weights
)

# ======================
# TRAIN
# ======================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop]
)

# ======================
# EVALUATE
# ======================
# Evaluate model on unseen test data
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)

# ======================
# SAVE MODEL
# ======================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        # Generate timestamp

model_path = f"models/cnn_model_{timestamp}.keras"      # Model save path
model.save(model_path)      # Save trained model

# ======================
# SAVE TRAINING HISTORY
# ======================
json_path = f"results/cnn_history_{timestamp}.json"

# Save training history as JSON
with open(json_path, "w") as f:
    json.dump(history.history, f)

# ======================
# SAVE RESULTS SUMMARY
# ======================
txt_path = f"results/cnn_results_{timestamp}.txt"
with open(txt_path, "w") as f:
    f.write("=== Custom CNN Results ===\n\n")

    for i in range(len(history.history['accuracy'])):
        f.write(f"Epoch {i+1}:\n")
        f.write(f"  Train Acc: {history.history['accuracy'][i]:.4f}\n")
        f.write(f"  Val Acc:   {history.history['val_accuracy'][i]:.4f}\n")
        f.write(f"  Train Loss:{history.history['loss'][i]:.4f}\n")
        f.write(f"  Val Loss:  {history.history['val_loss'][i]:.4f}\n")
        f.write(f"  Precision: {history.history['precision'][i]:.4f}\n")
        f.write(f"  Recall:    {history.history['recall'][i]:.4f}\n")
        f.write(f"  Val Precision: {history.history['val_precision'][i]:.4f}\n")
        f.write(f"  Val Recall:    {history.history['val_recall'][i]:.4f}\n\n")

    f.write(f"\nFinal Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Final Test Precision: {test_precision:.4f}\n")
    f.write(f"Final Test Recall: {test_recall:.4f}\n")


# ======================
# GRAPH - ACCURACY
# ======================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("CNN Accuracy")

acc_path = f"results/cnn_accuracy_{timestamp}.png"
plt.savefig(acc_path)
plt.close()


# ======================
# GRAPH - LOSS
# ======================
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("CNN Loss")

loss_path = f"results/cnn_loss_{timestamp}.png"
plt.savefig(loss_path)
plt.close()

# ======================
# FINAL OUTPUT
# ======================
print(f"\n✅ CNN Model saved: {model_path}")
print(f"✅ Results saved: {txt_path}")
print(f"✅ Accuracy graph saved: {acc_path}")
print(f"✅ Loss graph saved: {loss_path}")
print(f"✅ Test Accuracy: {test_acc:.4f}")
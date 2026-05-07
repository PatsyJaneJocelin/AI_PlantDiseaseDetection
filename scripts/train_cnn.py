import tensorflow as tf
from tensorflow.keras import layers, models
from load_data import get_data_generators

import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ======================
# LOAD DATA
# ======================
train_generator, val_generator, test_generator = get_data_generators(model_type="cnn")

# ======================
# BUILD MODEL
# ======================
def build_model():
    model = models.Sequential([
        tf.keras.Input(shape=(224,224,3)),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(3, activation='softmax')
    ])
    return model

model = build_model()

# ======================
# COMPILE MODEL
# ======================
model.compile(
    optimizer='adam',       # Adjusts weights to reduce error
    loss='categorical_crossentropy',        # Measures the error
    metrics=[                                   # Evaluates performance
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# ======================
# CALLBACKS
# ======================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
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
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)

# ======================
# SAVE MODEL
# ======================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"models/cnn_model_{timestamp}.keras"
model.save(model_path)

# ======================
# SAVE RESULTS
# ======================
json_path = f"results/cnn_history_{timestamp}.json"
with open(json_path, "w") as f:
    json.dump(history.history, f)

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


print(f"\n✅ CNN Model saved: {model_path}")
print(f"✅ Results saved: {txt_path}")
print(f"✅ Accuracy graph saved: {acc_path}")
print(f"✅ Loss graph saved: {loss_path}")
print(f"✅ Test Accuracy: {test_acc:.4f}")
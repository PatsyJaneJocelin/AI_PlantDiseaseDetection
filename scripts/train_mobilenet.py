import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from load_data import get_data_generators

import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ======================
# LOAD DATA (IMPORTANT)
# ======================
train_generator, val_generator, test_generator = get_data_generators(model_type="mobilenet")

# ======================
# BUILD MODEL
# ======================
def build_model():
    base_model = MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights='imagenet'
    )

    # base_model.trainable = False
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])

    return model

model = build_model()

# ======================
# COMPILE
# ======================
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
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

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_mobilenet.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# ======================
# CLASS WEIGHTS (Boost Early Blight)
# ======================
class_weight = {
    0: 1.3,  # Tomato___Early_blight
    1: 1.0,  # Tomato___Late_blight
    2: 1.0   # Tomato___healthy
}

# ======================
# TRAIN
# ======================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weight,   # 👈 ADD THIS
    callbacks=[early_stop, checkpoint]
)

# ======================
# EVALUATE
# ======================

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)

# ======================
# SAVE MODEL
# ======================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"models/mobilenet_model_{timestamp}.keras"
model.save(model_path)

# ======================
# SAVE RESULTS
# ======================

json_path = f"results/mobilenet_history_{timestamp}.json"
with open(json_path, "w") as f:

    json.dump(history.history, f)

txt_path = f"results/mobilenet_results_{timestamp}.txt"
with open(txt_path, "w") as f:
    f.write("=== MobileNetV2 Results ===\n\n")

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
plt.title("MobileNet Accuracy")

acc_path = f"results/mobilenet_accuracy_{timestamp}.png"
plt.savefig(acc_path)
plt.close()


# ======================
# GRAPH - LOSS
# ======================
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("MobileNet Loss")

loss_path = f"results/mobilenet_loss_{timestamp}.png"
plt.savefig(loss_path)
plt.close()


print(f"\n✅ MobileNet Model saved: {model_path}")
print(f"✅ Results saved: {txt_path}")
print(f"✅ Accuracy graph saved: {acc_path}")
print(f"✅ Loss graph saved: {loss_path}")
print(f"✅ Test Accuracy: {test_acc:.4f}")
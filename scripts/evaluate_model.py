# evaluate_model.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from load_data import get_data_generators
import os
from datetime import datetime

# ======================
# SETUP
# ======================
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================
# LOAD DATA
# ======================
_, _, test_generator = get_data_generators(model_type="mobilenet")

# ======================
# LOAD MODEL
# ======================
model = load_model("models/best_mobilenet.keras")

# ======================
# PREDICTIONS
# ======================
print("\n🔍 Running predictions on test data...")
preds = model.predict(test_generator)

y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# ======================
# LABELS
# ======================
labels = list(test_generator.class_indices.keys())
labels = sorted(labels, key=lambda x: test_generator.class_indices[x])

# ======================
# CONFUSION MATRIX (RAW)
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (Counts)")

cm_path = f"results/confusion_matrix_counts_{timestamp}.png"
plt.savefig(cm_path)
plt.close()

# ======================
# CONFUSION MATRIX (NORMALIZED)
# ======================
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6,6))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
disp_norm.plot(cmap=plt.cm.Blues, values_format='.2f')
plt.title("Confusion Matrix (Normalized)")

cm_norm_path = f"results/confusion_matrix_normalized_{timestamp}.png"
plt.savefig(cm_norm_path)
plt.close()

print(f"✅ Confusion matrices saved:")
print(f"   - {cm_path}")
print(f"   - {cm_norm_path}")

# ======================
# CLASSIFICATION REPORT
# ======================
report = classification_report(
    y_true,
    y_pred,
    target_names=labels
)

print("\n=== Classification Report ===\n")
print(report)

# Save report
report_path = f"results/classification_report_{timestamp}.txt"
with open(report_path, "w") as f:
    f.write(report)

print(f"✅ Classification report saved at: {report_path}")
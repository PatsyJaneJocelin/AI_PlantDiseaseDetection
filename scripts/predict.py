# predict.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil

# ======================
# SETUP
# ======================
os.makedirs("results", exist_ok=True)

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6

# ======================
# LOAD MODEL
# ======================
model = load_model("models/best_mobilenet.keras")

# ======================
# CLASS NAMES
# ======================
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

# ======================
# CLEAN LABEL
# ======================
def clean_label(label):
    return label.replace("___", " ").replace("_", " ")

# ======================
# RECOMMENDATIONS
# ======================
def get_recommendation(label):
    if "Early blight" in label:
        return "Use fungicide and remove infected leaves."
    elif "Late blight" in label:
        return "Apply copper-based fungicide immediately."
    elif "healthy" in label.lower():
        return "No action needed."
    else:
        return "Consult an agronomist."

# ======================
# PREDICT FUNCTION
# ======================
def predict(img_path):

    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocessing (MobileNet)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)[0]

    # Top prediction
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]
    prediction = clean_label(class_names[class_idx])

    print("\n======================")
    print("🌿 Final Prediction")
    print("======================")

    # Confidence check
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"⚠️ Low confidence ({confidence*100:.1f}%). Please retake the image.")
        return

    # Output
    if "healthy" in prediction.lower():
        print(f"✅ Plant appears healthy ({confidence*100:.1f}%)")
    else:
        print(f"⚠️ Disease detected: {prediction} ({confidence*100:.1f}%)")

    # Recommendation
    recommendation = get_recommendation(prediction)
    print(f"💡 Recommendation: {recommendation}")

    # ======================
    # TOP-2 PREDICTIONS
    # ======================
    sorted_indices = np.argsort(preds)[::-1]
    top_2 = sorted_indices[:2]

    print("\n🔍 Top Predictions:")
    for i in top_2:
        name = clean_label(class_names[i])
        print(f"{name}: {preds[i]*100:.1f}%")

    # ======================
    # SAVE RESULTS
    # ======================
    with open("results/predictions.txt", "a") as f:
        f.write(f"{img_path} → {prediction} ({confidence*100:.1f}%)\n")
        f.write("Top Predictions:\n")
        for i in top_2:
            name = clean_label(class_names[i])
            f.write(f"  {name}: {preds[i]*100:.1f}%\n")
        f.write(f"Recommendation: {recommendation}\n\n")

    # Save image copy
    try:
        filename = os.path.basename(img_path)
        shutil.copy(img_path, f"results/{filename}")
    except Exception as e:
        print("⚠️ Could not copy image:", e)

    print("\n✅ Prediction saved to results/")


# ======================
# RUN FROM TERMINAL
# ======================
if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()

    if not os.path.exists(img_path):
        print("❌ Image not found. Please check the path.")
    else:
        predict(img_path)
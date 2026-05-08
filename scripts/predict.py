import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil

# ======================
# SETUP
# ======================
os.makedirs("results", exist_ok=True)       # Create results folder if it doesn't exist

IMG_SIZE = (224, 224)       # Image size used during training
CONFIDENCE_THRESHOLD = 0.6      # Minimum confidence required for prediction

# ======================
# LOAD MODEL
# ======================
model = load_model("models/best_mobilenet.keras")       # Load trained MobileNet model

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
# Convert class name into readable format
def clean_label(label):
    return label.replace("___", " ").replace("_", " ")

# ======================
# RECOMMENDATIONS
# ======================
# Give treatment recommendation based on prediction
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

    img = image.load_img(img_path, target_size=IMG_SIZE)    # Resize image to model input size
    img_array = image.img_to_array(img)    # Convert image into array
    img_array = np.expand_dims(img_array, axis=0)    # Add extra dimension for model input

    # Applying MobileNet preprocessing
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_array = preprocess_input(img_array)

# ======================
# MODEL PREDICTION
# ======================
    # Get prediction probabilities
    preds = model.predict(img_array)[0]

    class_idx = np.argmax(preds)    # Get class with highest probability
    confidence = preds[class_idx]    # Confidence score
    prediction = clean_label(class_names[class_idx])    # Convert class label into readable text

    print("\n======================")
    print("🌿 Final Prediction")
    print("======================")

# ======================
# CONFIDENCE CHECK
# ======================
    # Reject uncertain predictions
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"⚠️ Low confidence ({confidence*100:.1f}%). Please retake the image.")
        return

    # ----------------------
    # DISPLAY RESULT
    # ----------------------
    # Healthy prediction
    if "healthy" in prediction.lower():
        print(f"✅ Plant appears healthy ({confidence*100:.1f}%)")
    # Disease prediction
    else:
        print(f"⚠️ Disease detected: {prediction} ({confidence*100:.1f}%)")

    # Show recommendation
    recommendation = get_recommendation(prediction)
    print(f"💡 Recommendation: {recommendation}")

    # ======================
    # TOP-2 PREDICTIONS
    # ======================
    sorted_indices = np.argsort(preds)[::-1]    # Sort predictions from highest to lowest
    top_2 = sorted_indices[:2]    # Take top 2 predictions

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

    # ----------------------
    # SAVE IMAGE COPY
    # ----------------------
    # Copy tested image into results folder
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
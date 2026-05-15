import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_my_model():
    return load_model("models/best_mobilenet.keras")

model = load_my_model()

# ======================
# CLASS NAMES
# ======================
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

IMG_SIZE = (224,224)

# ======================
# TITLE
# ======================
st.title("🌱 Crop Disease Detection")
st.write("Upload a tomato leaf image for disease prediction.")

# ======================
# FILE UPLOADER
# ======================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# ======================
# PREDICTION
# ======================
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)[0]

    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    prediction = class_names[class_idx]

    st.subheader("🔍 Prediction")

    if "Healthy" in prediction:
        st.success(f"✅ {prediction}")
    else:
        st.error(f"⚠️ {prediction}")

    st.write(f"Confidence: **{confidence*100:.2f}%**")

    # Top Predictions
    st.subheader("📊 Top Predictions")

    for i in np.argsort(preds)[::-1]:
        st.write(f"{class_names[i]}: {preds[i]*100:.2f}%")

    # Recommendation
    st.subheader("💡 Recommendation")

    if "Early Blight" in prediction:
        st.warning("Use fungicide and remove infected leaves.")
    elif "Late Blight" in prediction:
        st.warning("Apply copper-based fungicide immediately.")
    else:
        st.success("No action needed.")
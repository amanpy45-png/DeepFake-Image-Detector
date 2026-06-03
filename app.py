import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="DeepFake Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ AI DeepFake Image Detector")
st.write("Upload an image below to check if it is Real or an AI-generated Fake.")

# 2. Load the trained model safely
@st.cache_resource
def load_my_model():
    # Make sure this filename matches exactly what you saved
    return tf.keras.models.load_model("deepfake_detector_model.h5")

try:
    model = load_my_model()
except Exception as e:
    st.error("Could not load the model file. Make sure 'deepfake_detector_model.h5' is in the same folder as this script.")
    st.stop()

# 3. File Uploader Widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image on screen
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Analyzing image...")
    
    # 4. Preprocess the image exactly like training (64x64)
    # Convert image to RGB if it's RGBA (png)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize((64, 64))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)
    
    # 5. Run Prediction
    prediction_score = model.predict(img_batch)[0][0]
    
    # 6. Display Verdict
    st.markdown("---")
    if prediction_score >= 0.5:
        confidence = prediction_score * 100
        st.success(f"### Result: **REAL**")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        confidence = (1 - prediction_score) * 100
        st.error(f"### Result: **FAKE (DEEPFAKE)**")
        st.info(f"Confidence: {confidence:.2f}%")
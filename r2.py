import streamlit as st
import numpy as np
import cv2
import joblib
from pathlib import Path
import requests
from PIL import Image

# ===========================
# MODEL DOWNLOAD + LOAD
# ===========================

@st.cache_resource
def load_model():
    model_path = Path("deepfake_hybrid_model.pkl")

    if not model_path.exists():
        with st.spinner("📥 Downloading model from Google Drive..."):
            FILE_ID = "19TiXL0SSQViZy_fD_2TKA1t-6HRitzjc"
            URL = "https://drive.google.com/uc?export=download"

            session = requests.Session()
            response = session.get(URL, params={'id': FILE_ID}, stream=True)
            token = None

            # Detect Google Drive virus scan confirmation
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value

            if token:
                params = {'id': FILE_ID, 'confirm': token}
                response = session.get(URL, params=params, stream=True)

            # Save the actual binary content
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)

        st.success("✅ Model downloaded successfully!")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.warning("The downloaded file might be corrupted or incomplete.")
        st.stop()

    return model


# ===========================
# IMAGE PREPROCESSING
# ===========================

def preprocess_image(image):
    # Convert image to RGB
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.flatten().reshape(1, -1)
    return img_array


# ===========================
# STREAMLIT APP UI
# ===========================

st.set_page_config(page_title="Deepfake Image Detection", page_icon="🕵️", layout="centered")

st.title("🧠 Deepfake Image Detection App")
st.write("Upload an image and find out if it’s real or AI-generated!")

uploaded_file = st.file_uploader("📤 Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()

    if st.button("🔍 Analyze Image"):
        with st.spinner("Processing..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0]

        if prediction == 1:
            st.error("⚠️ This image appears to be a **Deepfake / AI-generated** image.")
        else:
            st.success("✅ This image appears to be **Real**.")


# ===========================
# FOOTER
# ===========================

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by Rohith Jabbireddy</p>",
    unsafe_allow_html=True
)

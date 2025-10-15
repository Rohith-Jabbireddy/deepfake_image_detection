import streamlit as st
import numpy as np
from PIL import Image
import joblib
import gdown
from pathlib import Path
import cv2
import plotly.graph_objects as go

# For Google Drive upload
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# -------------------------------
# 🌍 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="DeepFake Detective",
    page_icon="🧠",
    layout="wide",
)

# -------------------------------
# 🎨 STYLES & SMOOTH SCROLL
# -------------------------------
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        scroll-behavior: smooth;
        background: radial-gradient(circle at top left, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #8B5CF6;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        color: #BBB;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .card {
        background-color: #1F1B2E;
        border-radius: 15px;
        padding: 2rem;
        transition: transform .3s ease, box-shadow .3s ease;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
        text-align: center;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.6);
    }
    .nav-btn {
        background-color: #1F1B2E;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 5px;
        cursor: pointer;
        font-weight: 600;
        color: #8B5CF6;
        border: 1px solid #8B5CF6;
        text-decoration: none;
    }
    .nav-btn:hover {
        background-color: #8B5CF6;
        color: white;
    }
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#8B5CF6, #9333EA);
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧭 NAVIGATION BAR
# -------------------------------
st.markdown("""
    <div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 40px;">
        <a href='/?page=Home' class='nav-btn'>🏠 Home</a>
        <a href='/?page=Detect' class='nav-btn'>🔍 Detect</a>
        <a href='/?page=About' class='nav-btn'>📘 About</a>
        <a href='/?page=Contact' class='nav-btn'>📞 Contact</a>
    </div>
""", unsafe_allow_html=True)

# -------------------------------
# 🌐 PAGE ROUTING (using st.query_params)
# -------------------------------
params = st.query_params
page = params.get("page", ["Home"])[0]

# -------------------------------
# 🧩 LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model_path = Path("deepfake_hybrid_model.pkl")
    file_id = "19TiXL0SSQViZy_fD_2TKA1t-6HRitzjc"
    url = f"https://drive.google.com/uc?id={file_id}"
    if not model_path.exists():
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(url, str(model_path), quiet=False)
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.warning("Make sure the Google Drive file is public.")
        st.stop()

# -------------------------------
# 🔐 CONNECT TO GOOGLE DRIVE
# -------------------------------
@st.cache_resource
def connect_drive():
    gauth = GoogleAuth()
    # try load saved credentials
    gauth.LoadCredentialsFile("token.json")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("token.json")
    drive = GoogleDrive(gauth)
    return drive

# The target folder ID (from your shared link)
TARGET_DRIVE_FOLDER_ID = "1GJoX7tNDSfxLaYY24d7sBJwGBorpwq-4"

# -------------------------------
# 🏠 HOME PAGE
# -------------------------------
if page == "Home":
    st.markdown("<h1 class='main-title'>🧠 DeepFake Detective</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Harness the power of AI to detect DeepFakes in real-time</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h3>⚡ Real-Time Analysis</h3><p>Get instant AI results.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>🎯 High Accuracy</h3><p>Industry-level detection.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>🔒 Privacy First</h3><p>Your data stays local.</p></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("## 🔍 How It Works")
    st.markdown("""
    1️⃣ **Upload Image**  
    2️⃣ **AI Analysis**  
    3️⃣ **Heatmap Overlay & Confidence**  
    4️⃣ **Results Stored on Drive**
    """)

# -------------------------------
# 🔍 DETECTION PAGE
# -------------------------------
elif page == "Detect":
    st.title("🔍 DeepFake Detector")
    st.markdown("Upload an image to check if it's **Real or AI-generated**.")

    model = load_model()
    drive = connect_drive()

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        # Preprocess
        img_resized = image.resize((128, 128))
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        with st.spinner("🧠 AI is analyzing the image..."):
            prediction = model.predict(arr)
            confidence = float(np.max(prediction)) * 100
            predicted_label = np.argmax(prediction, axis=1)[0] if prediction.shape[1] > 1 else int(prediction[0] > 0.5)

            # Generate heatmap overlay
            gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
            blended = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)

            # Display results
            c1, c2 = st.columns(2)
            with c1:
                label_text = "🟢 Real Image" if predicted_label == 0 else "🔴 DeepFake Detected"
                st.markdown(f"## {label_text}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
            with c2:
                st.image(blended, caption="🔥 AI Heatmap Overlay", use_container_width=True)

            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence Level (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#8B5CF6"},
                    'steps': [
                        {'range': [0, 50], 'color': "#1E293B"},
                        {'range': [50, 100], 'color': "#9333EA"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # Save and upload files to Drive
        # -------------------------------
        try:
            save_dir = Path("analysis_uploads")
            save_dir.mkdir(exist_ok=True)
        except Exception:
            pass

        # Save original upload
        local_img_path = save_dir / uploaded_file.name
        image.save(local_img_path)

        # Save heatmap overlay (convert BGR->RGB for PIL)
        heatmap_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        heatmap_name = f"heatmap_{uploaded_file.name}"
        local_heatmap_path = save_dir / heatmap_name
        heatmap_img.save(local_heatmap_path)

        with st.spinner("☁️ Uploading to Google Drive..."):
            # original image
            file1 = drive.CreateFile({"title": uploaded_file.name,
                                      "parents": [{"id": TARGET_DRIVE_FOLDER_ID}]})
            file1.SetContentFile(str(local_img_path))
            file1.Upload()
            # heatmap file
            file2 = drive.CreateFile({"title": heatmap_name,
                                      "parents": [{"id": TARGET_DRIVE_FOLDER_ID}]})
            file2.SetContentFile(str(local_heatmap_path))
            file2.Upload()
        st.success("✅ Uploaded files to Google Drive folder")

# -------------------------------
# 📘 ABOUT PAGE
# -------------------------------
elif page == "About":
    st.title("📘 About DeepFake Detective")
    st.markdown("""
    DeepFake Detective uses a CNN to analyze images and highlight suspicious areas.

    **Features**  
    - Heatmap overlay  
    - Confidence score  
    - Automatic upload of results to your Google Drive  

    **Tech Stack**  
    - TensorFlow / Keras  
    - OpenCV, NumPy  
    - Streamlit  
    - PyDrive2 for Drive uploads  
    - Plotly for gauge  
    """)
    st.markdown("Built with ❤️ by **J Rohith Kumar Reddy**")

# -------------------------------
# 📞 CONTACT PAGE
# -------------------------------
elif page == "Contact":
    st.title("📞 Contact")
    st.markdown("""
    Got questions or collaboration ideas? Reach out:

    - 📧 Email: rohithjabbireddy@gmail.com  
    - 📱 Phone: +91 63049 43737  
    - LinkedIn: [linkedin.com/in/rohith-jabbireddy](https://www.linkedin.com/in/rohith-jabbireddy)
    """)

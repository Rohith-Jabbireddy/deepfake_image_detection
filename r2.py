import streamlit as st
import sqlite3
import cv2
import numpy as np
import joblib
from PIL import Image
from datetime import datetime
import hashlib
import io
import os
from pathlib import Path
import requests
import plotly.graph_objects as go

# =========================================================
# ✅ Download model safely from Google Drive
# =========================================================
@st.cache_resource
def load_model():
    model_path = Path("deepfake_hybrid_model.pkl")
    file_id = "19TiXL0SSQViZy_fD_2TKA1t-6HRitzjc"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if not model_path.exists():
        with st.spinner("📥 Downloading model from Google Drive..."):
            response = requests.get(download_url, stream=True)
            if response.status_code != 200:
                st.error(f"Download failed with status {response.status_code}")
                st.stop()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        if model_path.exists():
            model_path.unlink()
        st.stop()

model = load_model()

# =========================================================
# Database setup
# =========================================================
def init_db():
    conn = sqlite3.connect('deepfake_users.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, email TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS predictions 
                 (id INTEGER PRIMARY KEY, user_id INTEGER, timestamp TEXT, 
                 filename TEXT, result REAL, heatmap BLOB,
                 FOREIGN KEY(user_id) REFERENCES users(id))''')

    conn.commit()
    conn.close()

# =========================================================
# Utility functions
# =========================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def preprocess_image(image, target_size=(128, 128)):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img_bgr, target_size)
    normalized = resized.astype("float32") / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * normalized.mean(axis=2)), cv2.COLORMAP_JET)
    return np.expand_dims(normalized, axis=0), heatmap

# =========================================================
# App UI
# =========================================================
def main():
    init_db()
    st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="wide")

    # Gradient background
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(120deg, #141E30, #243B55);
            color: white;
        }
        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #00BFFF;
            text-shadow: 0px 0px 20px rgba(0,191,255,0.6);
        }
        .sub-title {
            text-align: center;
            font-size: 1.2rem;
            color: #B0C4DE;
            margin-bottom: 2rem;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 0 15px rgba(0,191,255,0.2);
        }
        .footer {
            text-align: center;
            color: #B0C4DE;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>🕵️ Deepfake Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Detect manipulated or AI-generated images in seconds</p>", unsafe_allow_html=True)

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.page = "Login"

    # =========================================================
    # LOGIN / SIGNUP
    # =========================================================
    if st.session_state.page == "Login" and not st.session_state.logged_in:
        with st.container():
            st.markdown("### 🔐 Login to Continue")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            if col1.button("Login"):
                conn = sqlite3.connect('deepfake_users.db')
                c = conn.cursor()
                c.execute("SELECT id, password FROM users WHERE username=?", (username,))
                result = c.fetchone()
                conn.close()
                if result and result[1] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.user_id = result[0]
                    st.session_state.page = "Detector"
                    st.success("Welcome back! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            if col2.button("Create Account"):
                st.session_state.page = "Signup"
                st.rerun()

    elif st.session_state.page == "Signup" and not st.session_state.logged_in:
        with st.container():
            st.markdown("### 🧾 Create Account")
            username = st.text_input("Choose Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Sign Up"):
                conn = sqlite3.connect('deepfake_users.db')
                c = conn.cursor()
                try:
                    c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                              (username, hash_password(password), email))
                    conn.commit()
                    st.success("✅ Account created! Please login.")
                    st.session_state.page = "Login"
                except sqlite3.IntegrityError:
                    st.error("Username already exists")
                conn.close()

    # =========================================================
    # DETECTOR PAGE
    # =========================================================
    elif st.session_state.page == "Detector" and st.session_state.logged_in:
        st.markdown("### 🔍 Upload Image for Deepfake Analysis")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🚀 Analyze Image"):
                with st.spinner("Analyzing... Please wait."):
                    image_data, heatmap = preprocess_image(image)
                    prediction = model.predict(image_data)[0].item()

                    status = "⚠️ Deepfake Detected" if prediction >= 0.55 else "✅ Real Image"
                    color = "#FF6347" if prediction >= 0.55 else "#00BFFF"

                    st.markdown(f"<div class='card'><h3 style='color:{color}'>{status}</h3>"
                                f"<p>Confidence: {prediction:.2%}</p></div>", unsafe_allow_html=True)
                    st.image(heatmap, caption="Detection Heatmap", width=300)

                    # Gauge visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction * 100,
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 55], 'color': '#00BFFF'},
                                {'range': [55, 100], 'color': '#FF6347'}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # Save to DB
                    conn = sqlite3.connect('deepfake_users.db')
                    c = conn.cursor()
                    heatmap_bytes = cv2.imencode('.png', heatmap)[1].tobytes()
                    c.execute("INSERT INTO predictions (user_id, timestamp, filename, result, heatmap) VALUES (?, ?, ?, ?, ?)",
                              (st.session_state.user_id, datetime.now().isoformat(), uploaded_file.name, prediction, heatmap_bytes))
                    conn.commit()
                    conn.close()

        if st.button("📜 View History"):
            st.session_state.page = "History"
            st.rerun()

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "Login"
            st.rerun()

    # =========================================================
    # HISTORY PAGE
    # =========================================================
    elif st.session_state.page == "History" and st.session_state.logged_in:
        st.markdown("### 📜 Analysis History")
        conn = sqlite3.connect('deepfake_users.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, filename, result, heatmap FROM predictions WHERE user_id=? ORDER BY timestamp DESC",
                  (st.session_state.user_id,))
        records = c.fetchall()
        conn.close()

        if not records:
            st.info("No analysis history yet.")
        else:
            for ts, fname, result, heatmap_bytes in records:
                status = "Deepfake" if result >= 0.55 else "Real"
                color = "#FF6347" if result >= 0.55 else "#00BFFF"
                with st.expander(f"{fname} - {status} ({result:.2%})"):
                    st.write(f"🕒 {ts}")
                    heatmap = cv2.imdecode(np.frombuffer(heatmap_bytes, np.uint8), cv2.IMREAD_COLOR)
                    st.image(heatmap, caption="Detection Heatmap", width=250)

        if st.button("⬅️ Back to Detector"):
            st.session_state.page = "Detector"
            st.rerun()

    # =========================================================
    # FOOTER
    # =========================================================
    st.markdown("""
        <div class='footer'>
            <p>© 2025 Deepfake Detector — Created by Rohith Jabbireddy</p>
            <p>📧 rohithjabbireddy@gmail.com | 📞 63049 43737</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

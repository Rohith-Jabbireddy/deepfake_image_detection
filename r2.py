import streamlit as st
import sqlite3
import cv2
import numpy as np
import joblib
from PIL import Image
from datetime import datetime
import hashlib
import base64
import io
import plotly.graph_objects as go
import os
from pathlib import Path
import gdown   # ✅ Added for Google Drive download

# =========================================================
# STEP 2: Download model from Google Drive if not present
# =========================================================
@st.cache_resource
def load_model():
    model_path = Path("deepfake_hybrid_model.pkl")
    if not model_path.exists():
        with st.spinner("📥 Downloading model from Google Drive..."):
            # 🔹 Replace YOUR_FILE_ID with your actual file ID from Google Drive
            url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
            gdown.download(url, str(model_path), quiet=False)
    return joblib.load(model_path)

model = load_model()
# =========================================================

# Database initialization with schema check
def init_db():
    conn = sqlite3.connect('deepfake_users.db')
    c = conn.cursor()
    
    # Create users table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, email TEXT)''')
    
    # Check if predictions table exists and has correct schema
    c.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in c.fetchall()]
    
    # If predictions table doesn't exist or is missing heatmap column, recreate it
    if 'heatmap' not in columns:
        c.execute("DROP TABLE IF EXISTS predictions")
        c.execute('''CREATE TABLE predictions 
                     (id INTEGER PRIMARY KEY, user_id INTEGER, timestamp TEXT, filename TEXT, result REAL, heatmap BLOB,
                     FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Preprocess image and generate heatmap
def preprocess_image(image, target_size=(128, 128)):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img_bgr, target_size)
    normalized = resized.astype("float32") / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * normalized.mean(axis=2)), cv2.COLORMAP_JET)
    return np.expand_dims(normalized, axis=0), heatmap

# Convert image to base64
def image_to_base64(img):
    buffered = io.BytesIO()
    Image.fromarray(img).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Main app
def main():
    init_db()
    
    # Page config
    st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="wide")
    
    # CSS for professional layout
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #1E2A44 0%, #2C3E50 50%, #34495E 100%);
            padding: 2rem;
            border-radius: 10px;
            min-height: 80vh;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
        }
        .nav-links a {
            color: #ECF0F1;
            text-decoration: none;
            margin-left: 1.5rem;
            font-weight: bold;
            transition: color 0.3s ease;
        }
        .nav-links a:hover {
            color: #00BFFF;
        }
        .stButton>button {
            background: linear-gradient(45deg, #00BFFF, #1E90FF);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #1E90FF, #00BFFF);
        }
        .title {
            color: #ECF0F1;
            font-size: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid #00BFFF;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .footer {
            background: #2C3E50;
            color: #ECF0F1;
            text-align: center;
            padding: 1rem;
            margin-top: 2rem;
            border-radius: 0 0 10px 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.page = "Login"

    # Header with navigation
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h1 class='title'>Deepfake Detection System</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                '<div class="nav-links">'
                '<a href="#" onclick="return false;">Home</a>'
                '<a href="#" onclick="return false;">Contact</a>'
                '<a href="#" onclick="return false;">About</a>'
                '</div>',
                unsafe_allow_html=True
            )

    # Main content area
    with st.container():
        if st.session_state.page == "Login" and not st.session_state.logged_in:
            st.markdown("<h2 style='text-align: center; color: #ECF0F1;'>Login</h2>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    conn = sqlite3.connect('deepfake_users.db')
                    c = conn.cursor()
                    c.execute("SELECT id, password FROM users WHERE username=?", (username,))
                    result = c.fetchone()
                    if result and result[1] == hash_password(password):
                        st.session_state.logged_in = True
                        st.session_state.user_id = result[0]
                        st.session_state.page = "Detector"
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                    conn.close()

        elif st.session_state.page == "Sign Up" and not st.session_state.logged_in:
            st.markdown("<h2 style='text-align: center; color: #ECF0F1;'>Sign Up</h2>", unsafe_allow_html=True)
            with st.form("register_form", clear_on_submit=True):
                username = st.text_input("Username", placeholder="Choose a username")
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Create a password")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    conn = sqlite3.connect('deepfake_users.db')
                    c = conn.cursor()
                    try:
                        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                                 (username, hash_password(password), email))
                        conn.commit()
                        st.success("Account created successfully! Please login.")
                        st.session_state.page = "Login"
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Username already exists")
                    conn.close()

        elif st.session_state.page == "Detector" and st.session_state.logged_in:
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader("Upload Image for Analysis", type=['jpg', 'jpeg', 'png'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if uploaded_file and st.button("Analyze Image"):
                    with st.spinner("Analyzing..."):
                        image_data, heatmap = preprocess_image(image)
                        prediction = model.predict(image_data)[0].item()
                        
                        # Store in database
                        conn = sqlite3.connect('deepfake_users.db')
                        c = conn.cursor()
                        heatmap_bytes = cv2.imencode('.png', heatmap)[1].tobytes()
                        c.execute("INSERT INTO predictions (user_id, timestamp, filename, result, heatmap) VALUES (?, ?, ?, ?, ?)",
                                 (st.session_state.user_id, datetime.now().isoformat(), uploaded_file.name, prediction, heatmap_bytes))
                        conn.commit()
                        conn.close()
                        
                        # Display result
                        status = "⚠️ Deepfake Detected" if prediction >= 0.55 else "✅ Real Image"
                        color = "#FF4500" if prediction >= 0.55 else "#00BFFF"
                        st.markdown(f"<div class='card'><h3 style='color: {color}'>{status}</h3><p>Confidence: {prediction:.2%}</p></div>", unsafe_allow_html=True)
                        st.image(heatmap, caption="Analysis Heatmap", width=300)
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={'axis': {'range': [0, 100]},
                                  'bar': {'color': color},
                                  'steps': [{'range': [0, 55], 'color': "#00BFFF"},
                                           {'range': [55, 100], 'color': "#FF4500"}]}))
                        st.plotly_chart(fig, use_container_width=True)

        elif st.session_state.page == "History" and st.session_state.logged_in:
            st.markdown("<h2 style='text-align: center; color: #ECF0F1;'>Analysis History</h2>", unsafe_allow_html=True)
            conn = sqlite3.connect('deepfake_users.db')
            c = conn.cursor()
            c.execute("SELECT timestamp, filename, result, heatmap FROM predictions WHERE user_id=? ORDER BY timestamp DESC",
                     (st.session_state.user_id,))
            history = c.fetchall()
            conn.close()
            
            for ts, fname, result, heatmap_bytes in history:
                status = "Deepfake" if result >= 0.55 else "Real"
                color = "#FF4500" if result >= 0.55 else "#00BFFF"
                with st.expander(f"{ts} - {fname}"):
                    st.markdown(f"<p style='color: {color}'>{status} ({result:.2%})</p>", unsafe_allow_html=True)
                    heatmap = cv2.imdecode(np.frombuffer(heatmap_bytes, np.uint8), cv2.IMREAD_COLOR)
                    st.image(heatmap, caption="Heatmap", width=200)

    # Navigation buttons
    nav_cols = st.columns(3 if st.session_state.logged_in else 2)
    if st.session_state.logged_in:
        with nav_cols[0]:
            if st.button("Detector", key="detector_btn"):
                st.session_state.page = "Detector"
                st.rerun()
        with nav_cols[1]:
            if st.button("History", key="history_btn"):
                st.session_state.page = "History"
                st.rerun()
        with nav_cols[2]:
            if st.button("Logout", key="logout_btn"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.page = "Login"
                st.success("Logged out successfully!")
                st.rerun()
    else:
        with nav_cols[0]:
            if st.button("Login", key="login_btn"):
                st.session_state.page = "Login"
                st.rerun()
        with nav_cols[1]:
            if st.button("Sign Up", key="signup_btn"):
                st.session_state.page = "Sign Up"
                st.rerun()

    # Footer
    st.markdown("""
        <div class='footer'>
            <p>© 2025 Deepfake Detector Inc. All rights reserved.</p>
            <p>Contact us: Rohithjabbireddy@gmail.com | Phone: 6304943737</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

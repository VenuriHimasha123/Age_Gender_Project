import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import tensorflow as tf
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NEON-AI | Demographics", layout="wide", page_icon="⚡")

# --- 1. PROFESSIONAL GRADIENT & NEON CSS ---
st.markdown("""
    <style>
    /* Gradient Background for the main app */
    .stApp {
        background: linear-gradient(135deg, #2e2e2e 0%, #ff4b1f 100%);
        background-attachment: fixed;
    }
    
    /* Neon Yellow Text and Glow for all elements */
    h1, h2, h3, h4, h5, h6, .stText, p, label, .stMarkdown {
        color: #EAFF00 !important;
        text-shadow: 0 0 10px #EAFF00, 0 0 20px #EAFF00;
        font-family: 'Courier New', Courier, monospace;
    }

    /* Sidebar Styling - Dark with Neon Border */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        border-right: 2px solid #EAFF00;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #EAFF00 !important;
        text-shadow: 0 0 5px #EAFF00;
    }
    [data-testid="stMetricLabel"] p {
        color: #EAFF00 !important;
    }

    /* File Uploader Box */
    .stFileUploader {
        border: 2px dashed #EAFF00;
        padding: 10px;
        border-radius: 15px;
        background-color: rgba(0, 0, 0, 0.4);
    }

    /* Success/Warning/Error boxes to match Neon */
    .stAlert {
        background-color: rgba(0, 0, 0, 0.6);
        color: #EAFF00 !important;
        border: 1px solid #EAFF00;
    }
    
    /* Button and Widget Glow */
    .stButton>button {
        background-color: #000000;
        color: #EAFF00;
        border: 2px solid #EAFF00;
        box-shadow: 0 0 10px #EAFF00;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("# ⚡ NEON-AI")
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Analysis", "Dataset", "Settings"],
        icons=["house", "camera", "database", "gear"],
        menu_icon="cast",
        default_index=1,
        styles={
            "container": {"background-color": "#000", "padding": "5px"},
            "icon": {"color": "#EAFF00", "font-size": "20px"}, 
            "nav-link": {"color": "#EAFF00", "font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#EAFF00", "color": "black", "font-weight": "bold"},
        }
    )
    st.markdown("---")
    st.info("Group 09 | KDU Undergrad")

# --- 3. PAGE LOGIC ---

if selected == "Dashboard":
    st.title("🚀 SYSTEM OVERVIEW")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", "ONLINE")
    col2.metric("Accuracy", "95.9%", delta="Gender")
    col3.metric("Backend", "TF/Keras")
    st.markdown("---")
    st.write("This intelligence system provides real-time demographic estimation using Deep Learning.")

elif selected == "Analysis":
    st.title("🔍 FACIAL ANALYSIS ENGINE")
    
    cascade_path = "models/haarcascade_frontalface_default.xml"
    model_path = "models/gender_model.h5" 
    
    if not os.path.exists(cascade_path) or not os.path.exists(model_path):
        st.error("❌ CRITICAL: Missing model files in 'models/' folder.")
    else:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gender_model = tf.keras.models.load_model(model_path)
        
        uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            display_img = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Preprocessing
                    face_crop = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_crop, (128, 128))
                    face_normalized = face_resized / 255.0
                    face_reshaped = np.reshape(face_normalized, (1, 128, 128, 3))
                    
                    # Inference
                    prediction = gender_model.predict(face_reshaped)
                    gender = "FEMALE" if prediction[0][0] > 0.5 else "MALE"
                    prob = prediction[0][0] if gender == "FEMALE" else 1 - prediction[0][0]
                    
                    # Drawing
                    color = (0, 255, 234) # Cyan/Yellow variant for visibility
                    label = f"{gender} ({prob*100:.1f}%)"
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 255), 4)
                    cv2.putText(display_img, label, (x, y-15), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                
                st.image(display_img, channels="BGR", use_container_width=True)
                st.success(f"⚡ Analysis Complete: Identified {len(faces)} subject(s).")
            else:
                st.warning("⚠️ Face localized, but features unclear. Try another photo.")

elif selected == "Dataset":
    st.title("📊 DATASET REPOSITORY")
    st.markdown("""
    ### UTKFace Database
    - **Capacity:** 23,705 high-resolution images.
    - **Labels:** Age, Gender, and Ethnicity.
    - **Preprocessing:** All images resized to 128x128 for CNN training.
    """)

elif selected == "Settings":
    st.title("⚙️ SYSTEM SETTINGS")
    st.write("UI Theme: Neo Grey-Orange Gradient")
    st.write("Model Type: Convolutional Neural Network (CNN)")
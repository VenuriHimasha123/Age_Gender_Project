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
    .stApp {
        background: linear-gradient(135deg, #2e2e2e 0%, #ff4b1f 100%);
        background-attachment: fixed;
    }
    h1, h2, h3, h4, h5, h6, .stText, p, label, span, .stMarkdown {
        color: #EAFF00 !important;
        text-shadow: 0 0 10px #EAFF00, 0 0 20px #EAFF00;
        font-family: 'Courier New', Courier, monospace;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        border-right: 2px solid #EAFF00;
    }
    [data-testid="stMetricValue"] {
        color: #EAFF00 !important;
        text-shadow: 0 0 5px #EAFF00;
    }
    .stFileUploader, .stCamera {
        border: 2px dashed #EAFF00;
        padding: 10px;
        border-radius: 15px;
        background-color: rgba(0, 0, 0, 0.4);
    }
    .stAlert {
        background-color: rgba(0, 0, 0, 0.6);
        color: #EAFF00 !important;
        border: 1px solid #EAFF00;
    }
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
    col2.metric("Gender AI", "95.9%", delta="Verified")
    col3.metric("Age AI", "MAE < 10", delta="Regression")
    st.markdown("---")
    st.write("Current Phase: Multi-Task Inference (Age + Gender)")

elif selected == "Analysis":
    st.title("🔍 FACIAL ANALYSIS ENGINE")
    
    cascade_path = "models/haarcascade_frontalface_default.xml"
    # Note: Using the combined model name we saved in the new train_cnn.py
    model_path = "models/age_gender_model.h5" 
    
    if not os.path.exists(cascade_path) or not os.path.exists(model_path):
        st.error("❌ CRITICAL: Missing model files in 'models/' folder.")
        st.warning("Ensure 'age_gender_model.h5' exists after running the new train_cnn.py")
    else:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        # Load the multi-output model
        multi_model = tf.keras.models.load_model(model_path, compile=False)
        
        mode = st.radio("Select Input Mode:", ["Upload Image", "Live Webcam"], horizontal=True)
        
        input_image = None
        if mode == "Upload Image":
            uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                input_image = cv2.imdecode(file_bytes, 1)
        else:
            cam_input = st.camera_input("Take a snapshot")
            if cam_input:
                file_bytes = np.asarray(bytearray(cam_input.read()), dtype=np.uint8)
                input_image = cv2.imdecode(file_bytes, 1)

        if input_image is not None:
            display_img = input_image.copy()
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # 1. Preprocessing (128x128 as per training)
                    face_crop = input_image[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_crop, (128, 128))
                    face_normalized = face_resized / 255.0
                    face_reshaped = np.reshape(face_normalized, (1, 128, 128, 3))
                    
                    # 2. Multi-Task Inference
                    predictions = multi_model.predict(face_reshaped)
                    
                    # predictions[0] is Gender Output, predictions[1] is Age Output
                    gender_pred = predictions[0][0][0]
                    age_pred = int(predictions[1][0][0])
                    
                    gender_label = "FEMALE" if gender_pred > 0.5 else "MALE"
                    prob = gender_pred if gender_label == "FEMALE" else 1 - gender_pred
                    
                    # 3. Combined Result Label
                    result_text = f"{gender_label}, ~{age_pred} yrs ({prob*100:.1f}%)"
                    
                    # 4. Drawing with Neon Yellow
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 255), 4)
                    cv2.putText(display_img, result_text, (x, y-15), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                
                st.image(display_img, channels="BGR", use_container_width=True)
                st.success(f"⚡ Multi-Task Analysis Complete: Found {len(faces)} subject(s).")
            else:
                st.warning("⚠️ No faces detected. Ensure lighting is sufficient.")

elif selected == "Dataset":
    st.title("📊 DATASET REPOSITORY")
    st.markdown("""
    ### UTKFace Database
    - **Total Records:** 23,705 images.
    - **Multi-Task Labels:** Gender (Classification) & Age (Regression).
    - **Target Size:** 128x128 pixels (RGB).
    """)

elif selected == "Settings":
    st.title("⚙️ SYSTEM SETTINGS")
    st.write("UI Theme: Neo Grey-Orange Gradient")
    st.write("Engine: Dual-Output CNN (Functional API)")
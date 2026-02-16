import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import tensorflow as tf
import os
from pymongo import MongoClient
import certifi

# --- 1. MONGODB CONNECTION SETTINGS ---
# Using your verified credentials
MONGO_URI = "mongodb+srv://venurihimasha123_db_user:venuri@cluster0.uhy55kg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

@st.cache_resource
def init_connection():
    try:
        # certifi handles SSL issues; serverSelectionTimeout prevents the app from hanging
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Verify connection
        return client.AgeGenderDB        # Database Name
    except Exception as e:
        return None

db = init_connection()

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="NEON-AI | Secure Access", layout="wide", page_icon="⚡")

# --- 3. PROFESSIONAL GRADIENT & NEON CSS ---
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
    .stTextInput>div>div>input {
        background-color: #000000 !important;
        color: #EAFF00 !important;
        border: 1px solid #EAFF00 !important;
    }
    .stButton>button {
        background-color: #000000;
        color: #EAFF00;
        border: 2px solid #EAFF00;
        box-shadow: 0 0 10px #EAFF00;
        width: 100%;
    }
    /* Style for the tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #000; border-radius: 4px; border: 1px solid #EAFF00; color: #EAFF00; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_fullname" not in st.session_state:
    st.session_state.user_fullname = ""

# --- 5. AUTHENTICATION & SYSTEM LOGIC ---

# CHECK IF DATABASE IS ONLINE
if db is None:
    st.error("❌ DATABASE OFFLINE: Please check your Internet or MongoDB Network Access (IP Whitelist).")
    st.info("Check your terminal/CMD for the technical error details.")

# SHOW LOGIN/REGISTER IF NOT LOGGED IN
elif not st.session_state.logged_in:
    st.title("⚡ NEON-AI ACCESS CONTROL")
    tab1, tab2 = st.tabs(["🔑 LOGIN", "📝 REGISTER"])

    with tab1:
        login_user = st.text_input("Username", key="login_u")
        login_pass = st.text_input("Password", type="password", key="login_p")
        if st.button("Login"):
            user = db.users.find_one({"username": login_user, "password": login_pass})
            if user:
                st.session_state.logged_in = True
                st.session_state.user_fullname = user["name"]
                st.rerun()
            else:
                st.error("Invalid Username or Password")

    with tab2:
        reg_name = st.text_input("Full Name")
        reg_user = st.text_input("New Username")
        reg_email = st.text_input("Email")
        reg_pass = st.text_input("Password", type="password")
        reg_conf = st.text_input("Confirm Password", type="password")
        
        if st.button("Create Account"):
            if reg_pass != reg_conf:
                st.error("Passwords do not match!")
            elif db.users.find_one({"username": reg_user}):
                st.error("Username already exists!")
            elif reg_name and reg_user and reg_pass:
                db.users.insert_one({
                    "name": reg_name,
                    "username": reg_user,
                    "email": reg_email,
                    "password": reg_pass
                })
                st.success("Registration Successful! Please switch to the Login tab.")
            else:
                st.error("Please fill in all fields.")

# --- 6. MAIN SYSTEM (ONLY ACCESSIBLE AFTER LOGIN) ---
else:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user_fullname}")
        selected = option_menu(
            menu_title="Main Menu",
            options=["Dashboard", "Analysis", "Dataset", "Settings"],
            icons=["house", "camera", "database", "gear"],
            menu_icon="cast", default_index=1,
            styles={
                "nav-link-selected": {"background-color": "#EAFF00", "color": "black"},
            }
        )
        st.markdown("---")
        if st.button("🔴 LOGOUT"):
            st.session_state.logged_in = False
            st.session_state.user_fullname = ""
            st.rerun()

    if selected == "Dashboard":
        st.title("🚀 SYSTEM OVERVIEW")
        st.subheader(f"Welcome, {st.session_state.user_fullname}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", "ONLINE", delta="Cloud Active")
        col2.metric("Gender AI", "95.9%", delta="Verified")
        col3.metric("Age AI", "MAE < 10", delta="Regression")
        st.markdown("---")
        st.write("Current Phase: Multi-Task Inference (Age + Gender)")

    elif selected == "Analysis":
        st.title("🔍 FACIAL ANALYSIS ENGINE")
        
        cascade_path = "models/haarcascade_frontalface_default.xml"
        model_path = "models/age_gender_model.h5" 
        
        if not os.path.exists(cascade_path) or not os.path.exists(model_path):
            st.error("❌ CRITICAL: Missing model files in 'models/' folder.")
        else:
            face_cascade = cv2.CascadeClassifier(cascade_path)
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
                        # 1. Preprocessing (128x128)
                        face_crop = input_image[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_crop, (128, 128))
                        face_norm = face_resized / 255.0
                        face_reshaped = np.reshape(face_norm, (1, 128, 128, 3))
                        
                        # 2. Multi-Task Inference
                        predictions = multi_model.predict(face_reshaped)
                        gender_pred = predictions[0][0][0]
                        age_pred = int(predictions[1][0][0])
                        
                        gender_label = "FEMALE" if gender_pred > 0.5 else "MALE"
                        prob = gender_pred if gender_label == "FEMALE" else 1 - gender_pred
                        
                        # 3. Draw Results
                        result_text = f"{gender_label}, ~{age_pred} yrs ({prob*100:.1f}%)"
                        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 255), 4)
                        cv2.putText(display_img, result_text, (x, y-15), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                    
                    st.image(display_img, channels="BGR", use_container_width=True)
                    st.success(f"Analysis Complete: Found {len(faces)} subject(s).")
                else:
                    st.warning("⚠️ No faces detected.")

    elif selected == "Dataset":
        st.title("📊 DATASET REPOSITORY")
        st.markdown("### UTKFace Database")
        st.write("- 23,705 images labels with Age and Gender.")

    elif selected == "Settings":
        st.title("⚙️ SYSTEM SETTINGS")
        st.write("Engine: Dual-Output CNN (Functional API)")
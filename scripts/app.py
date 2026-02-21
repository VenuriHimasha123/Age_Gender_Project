import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import tensorflow as tf
import os
from pymongo import MongoClient
import certifi
from datetime import datetime
from fpdf import FPDF
import tempfile
import uuid

# --- 1. MONGODB CONNECTION SETTINGS ---
MONGO_URI = "mongodb+srv://venurihimasha123_db_user:venuri@cluster0.uhy55kg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

@st.cache_resource
def init_connection():
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client.AgeGenderDB
    except Exception:
        return None

db = init_connection()

# --- 2. PDF GENERATION LOGIC (FIXED FOR WINDOWS + CROPPED IMAGES) ---
class BiometricReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(46, 46, 46)
        self.cell(0, 10, 'CHRONOSID ANALYTICS - BIOMETRIC PROFILE', 0, 1, 'C')
        self.ln(5)

def generate_pdf_report(operator, results):
    pdf = BiometricReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Metadata
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Authorized Operator: {operator}", ln=True)
    pdf.cell(0, 10, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Total Subjects Detected: {len(results)}", ln=True)
    pdf.ln(10)

    # Table Header
    pdf.set_fill_color(234, 255, 0) 
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Profile", 1, 0, 'C', True)
    pdf.cell(35, 10, "Subject ID", 1, 0, 'C', True)
    pdf.cell(35, 10, "Gender", 1, 0, 'C', True)
    pdf.cell(40, 10, "Age Range", 1, 0, 'C', True)
    pdf.cell(40, 10, "Confidence", 1, 1, 'C', True)

    pdf.set_font("Arial", size=10)
    
    temp_image_paths = []

    for res in results:
        # Create a unique temporary path for each crop to avoid PermissionErrors on Windows
        temp_path = os.path.join(tempfile.gettempdir(), f"crop_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(temp_path, res['crop_bgr'])
        temp_image_paths.append(temp_path)

        curr_y = pdf.get_y()
        # Add image to PDF cell (placed within the first column)
        pdf.image(temp_path, x=13, y=curr_y + 2, w=34, h=16)
        
        # Table Row (Empty string for the first cell where the image is placed)
        pdf.cell(40, 20, "", 1) 
        pdf.cell(35, 20, res['id'], 1, 0, 'C')
        pdf.cell(35, 20, res['gender'], 1, 0, 'C')
        pdf.cell(40, 20, f"{res['age_range']} Yrs", 1, 0, 'C')
        pdf.cell(40, 20, f"{res['confidence']}%", 1, 1, 'C')

    # Output to byte stream
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    # Cleanup temp files after generation is finished
    for path in temp_image_paths:
        try:
            os.remove(path)
        except:
            pass

    return pdf_output

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="ChronosID Analytics | Secure Vision", layout="wide", page_icon="🛡️")

# --- 4. ADVANCED NEON, 3D CARDS & ANIMATION CSS ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #2e2e2e 0%, #1a1a1a 100%); background-attachment: fixed; }
    h1, h2, h3, h4, h5, h6, .stText, p, label, span, .stMarkdown {
        color: #EAFF00 !important; text-shadow: 0 0 15px #EAFF00; font-family: 'Courier New', Courier, monospace;
    }
    .typewriter-text {
        overflow: hidden; border-right: .15em solid #EAFF00; white-space: nowrap; margin: 0 auto; letter-spacing: .10em;
        font-size: 1.1rem; font-weight: bold; animation: typing 5s steps(50, end), blink-caret .75s step-end infinite;
    }
    @keyframes typing { from { width: 0 } to { width: 100% } }
    @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: #EAFF00; } }

    .metric-card, .auth-card {
        background: rgba(0,0,0,0.7); border: 2px solid #EAFF00; border-radius: 15px; padding: 25px;
        box-shadow: 0 10px 30px rgba(234, 255, 0, 0.2); transition: transform 0.3s;
    }
    .metric-card:hover { transform: translateY(-5px); }

    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 2px solid #EAFF00; }
    .stButton>button {
        background-color: #000; color: #EAFF00; border: 2px solid #EAFF00;
        box-shadow: 0 0 10px #EAFF00; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #EAFF00; color: #000; }
    
    .scanner-container { position: relative; overflow: hidden; border: 2px solid #EAFF00; border-radius: 10px; }
    .scanner-line {
        position: absolute; width: 100%; height: 4px; background: #EAFF00;
        box-shadow: 0 0 15px #EAFF00; animation: scan 3s linear infinite; z-index: 10;
    }
    @keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }
    </style>
    """, unsafe_allow_html=True)

# --- 5. SESSION STATE ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_fullname" not in st.session_state: st.session_state.user_fullname = ""

# --- 6. AUTHENTICATION / SIDEBAR ---
if db is None:
    st.error("❌ DATABASE OFFLINE")
    st.stop()

with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.user_fullname if st.session_state.logged_in else 'GUEST MODE'}")
    selected = option_menu("ChronosID Menu", ["Dashboard", "Analysis", "Dataset"], 
                           icons=["house", "camera", "database"], menu_icon="shield-shaded", default_index=0,
                           styles={"nav-link-selected": {"background-color": "#EAFF00", "color": "black"}})
    if st.session_state.logged_in and st.button("🔴 LOGOUT"):
        st.session_state.logged_in = False
        st.rerun()

# --- 7. PAGE LOGIC ---
if selected == "Dashboard":
    st.title("🛡️ CHRONOSID ANALYTICS")
    st.markdown('<div class="typewriter-text">Advanced face and gender estimation for secure biometric profiling.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1.2])
    with col1:
        st.markdown("### 🚀 ENGINE OPERATIONAL STATUS")
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown('<div class="metric-card"> CLOUD DB<br><b>CONNECTED</b></div>', unsafe_allow_html=True)
        with m2: st.markdown('<div class="metric-card"> AI CORE<br><b>OPTIMIZED</b></div>', unsafe_allow_html=True)
        with m3: st.markdown('<div class="metric-card"> SECURITY<br><b>ENFORCED</b></div>', unsafe_allow_html=True)
    with col2:
        if not st.session_state.logged_in:
            st.markdown('<div class="auth-card">### 🔐 ACCESS PORTAL', unsafe_allow_html=True)
            t1, t2 = st.tabs(["LOGIN", "REGISTER"])
            with t1:
                u, p = st.text_input("Username"), st.text_input("Password", type="password")
                if st.button("AUTHENTICATE"):
                    user = db.users.find_one({"username": u, "password": p})
                    if user:
                        st.session_state.logged_in, st.session_state.user_fullname = True, user["name"]
                        st.rerun()
            with t2:
                rn, ru, rp = st.text_input("Full Name"), st.text_input("New User"), st.text_input("New Pass", type="password")
                if st.button("CREATE PROFILE"):
                    db.users.insert_one({"name": rn, "username": ru, "password": rp})
                    st.success("Saved!")
            st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Analysis":
    st.title("🔍 CHRONOSID VISION ENGINE")
    if not st.session_state.logged_in:
        st.warning("🔒 SECURE ACCESS REQUIRED.")
    else:
        c_path, m_path = "models/haarcascade_frontalface_default.xml", "models/age_gender_model.h5"
        face_cascade = cv2.CascadeClassifier(c_path)
        multi_model = tf.keras.models.load_model(m_path, compile=False)
        
        mode = st.radio("Select Biometric Source:", ["Static Upload", "Live Capture"], horizontal=True)
        input_image = None
        
        if mode == "Static Upload":
            up = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
            if up: input_image = cv2.imdecode(np.asarray(bytearray(up.read()), dtype=np.uint8), 1)
        else:
            cam = st.camera_input("Scanner Active")
            if cam: input_image = cv2.imdecode(np.asarray(bytearray(cam.read()), dtype=np.uint8), 1)

        if input_image is not None:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                st.error("🛑 BIOMETRIC DATA NOT RECOGNIZED")
            else:
                display_img = input_image.copy()
                analysis_results = []
                
                st.markdown(f"### 📡 SUBJECT TELEMETRY (Detected: {len(faces)})")
                cols = st.columns(len(faces))

                for i, (x, y, w, h) in enumerate(faces):
                    roi = input_image[y:y+h, x:x+w]
                    f_crop = cv2.resize(roi, (128, 128)) / 255.0
                    preds = multi_model.predict(np.reshape(f_crop, (1, 128, 128, 3)))
                    
                    # --- GENDER LOGIC ---
                    g_prob = preds[0][0][0]
                    # Change: Now 0.0-0.5 is MALE and 0.5-1.0 is FEMALE
                    g_lab = "FEMALE" if g_prob > 0.5 else "MALE"
                    base_age = int(preds[1][0][0])
                    # Ensure age range doesn't drop below 0
                    age_range = f"{max(0, base_age - 2)}-{base_age + 2}"
                    acc = round((g_prob if g_lab == "FEMALE" else 1 - g_prob) * 100, 1)
                    
                    analysis_results.append({
                        'id': f'SUB_{i+1}', 
                        'gender': g_lab, 
                        'age_range': age_range, 
                        'confidence': acc, 
                        'crop_bgr': roi
                    })

                    # Draw on Main Image
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 255), 4)
                    cv2.putText(display_img, f"SUB_{i+1}", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

                    with cols[i]:
                        st.markdown(f"**SUBJECT {i+1}**")
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        st.image(roi_rgb, caption=f"Bio-Profile {i+1}", use_container_width=True)
                        st.write(f"🧬 **Gender:** {g_lab}")
                        st.write(f"⏳ **Range:** {age_range} Yrs")
                        st.write(f"🎯 **Confidence:** {acc}%")

                st.markdown("---")
                st.markdown('<div class="scanner-container"><div class="scanner-line"></div>', unsafe_allow_html=True)
                st.image(display_img, channels="BGR", caption="Full Biometric Scan Overlay", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # PDF Button
                pdf_data = generate_pdf_report(st.session_state.user_fullname, analysis_results)
                st.download_button(label="📥 DOWNLOAD BIOMETRIC REPORT (PDF)", data=pdf_data, 
                                   file_name=f"ChronosID_Report_{datetime.now().strftime('%H%M%S')}.pdf", mime="application/pdf")
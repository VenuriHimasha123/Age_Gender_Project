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

import base64

# Add this function here
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Replace this path with the path to your image on your desktop
# Example: "C:/Users/YourName/Desktop/background.jpg"
# IMPORTANT: Use forward slashes (/) or double backslashes (\\)
bin_str = get_base64("assets/background.png")

# --- 1. MONGODB CONNECTION SETTINGS ---
MONGO_URI = "mongodb+srv://venurihimasha123_db_user:venuri@cluster0.uhy55kg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

@st.cache_resource
def init_connection():
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        return client.AgeGenderDB
    except Exception:
        return None

db = init_connection()

# --- 2. PDF GENERATION LOGIC ---
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
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Authorized Operator: {operator}", ln=True)
    pdf.cell(0, 10, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Total Subjects Detected: {len(results)}", ln=True)
    pdf.ln(10)
    pdf.set_fill_color(234, 255, 0) 
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Profile", 1, 0, 'C', True)
    pdf.cell(35, 10, "ID", 1, 0, 'C', True)
    pdf.cell(35, 10, "Gender", 1, 0, 'C', True)
    pdf.cell(40, 10, "Age Range", 1, 0, 'C', True)
    pdf.cell(40, 10, "Confidence", 1, 1, 'C', True)
    
    temp_image_paths = []
    for res in results:
        temp_path = os.path.join(tempfile.gettempdir(), f"crop_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(temp_path, res['crop_bgr'])
        temp_image_paths.append(temp_path)
        curr_y = pdf.get_y()
        pdf.image(temp_path, x=13, y=curr_y + 2, w=34, h=16)
        pdf.cell(40, 20, "", 1) 
        pdf.cell(35, 20, res['id'], 1, 0, 'C')
        pdf.cell(35, 20, res['gender'], 1, 0, 'C')
        pdf.cell(40, 20, f"{res['age_range']} Yrs", 1, 0, 'C')
        pdf.cell(40, 20, f"{res['confidence']}%", 1, 1, 'C')
    
    pdf_output = pdf.output(dest='S').encode('latin-1')
    for path in temp_image_paths:
        try: os.remove(path)
        except: pass
    return pdf_output

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="ChronosID Analytics | Hyper Vision", layout="wide", page_icon="")

# --- 4. HD BACKGROUND, GLASSMORPHISM & SMOOTH SCROLLING ---
# Replace the URL below with any high-quality image link you prefer
bg_image_url = "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop"

# --- 4. ADVANCED CSS: GLASSMORPHISM, SMOOTH SCROLLING & UI FIXES ---
st.markdown(f"""
    <style>
    /* Global Smooth Scrolling */
    html {{ 
        scroll-behavior: smooth; 
    }}
    
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                          url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    
    }}

    /* GLASSMORPHISM UNIVERSAL STYLE */
    .metric-card, .auth-card, .stTabs, [data-testid="stMetricValue"], 
    [data-testid="stSidebar"] > div:first-child, .stFileUploader, .stCameraInput {{
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(25px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6) !important;
        transition: transform 0.3s ease;
        padding: 20px;
        margin-bottom: 10px;
    }}

    [data-testid="stSidebar"] {{
        background-color: transparent !important;
        border-right: 1px solid rgba(0, 242, 255, 0.2);
    }}

    /* 3D Neural Cube Component */
    .single-cube-container {{
        width: 100%; height: 280px; display: flex; justify-content: center; align-items: center;
        perspective: 1000px; margin: 20px 0;
    }}
    .cube {{
        width: 100px; height: 100px; position: relative; transform-style: preserve-3d;
        animation: rotateFull 12s infinite linear;
    }}
    .face {{
        position: absolute; width: 100px; height: 100px; border: 2px solid #00f2ff;
        background: rgba(0, 242, 255, 0.1); box-shadow: 0 0 25px #00f2ff, inset 0 0 15px #00f2ff;
    }}
    .front  {{ transform: rotateY(0deg) translateZ(50px); }}
    .back   {{ transform: rotateY(180deg) translateZ(50px); }}
    .right  {{ transform: rotateY(90deg) translateZ(50px); }}
    .left   {{ transform: rotateY(-90deg) translateZ(50px); }}
    .top    {{ transform: rotateX(90deg) translateZ(50px); }}
    .bottom {{ transform: rotateX(-90deg) translateZ(50px); }}

    @keyframes rotateFull {{
        from {{ transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg); }}
        to {{ transform: rotateX(360deg) rotateY(360deg) rotateZ(360deg); }}
    }}

    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {{
        color: #00f2ff !important; text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
        font-family: 'Courier New', Courier, monospace;
    }}

    .stButton>button {{
        background: rgba(0, 242, 255, 0.1) !important;
        color: #00f2ff !important; border: 1px solid #00f2ff !important;
        border-radius: 12px !important; backdrop-filter: blur(10px);
        transition: 0.4s;
    }}
    .stButton>button:hover {{ 
        background: #00f2ff !important; color: black !important; box-shadow: 0 0 30px #00f2ff; 
    }}
    
    input {{ 
        background: rgba(255, 255, 255, 0.05) !important; 
        color: white !important; 
        border: 1px solid rgba(0, 242, 255, 0.3) !important;
        border-radius: 10px !important; 
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. SESSION STATE ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_fullname" not in st.session_state: st.session_state.user_fullname = ""

# --- 6. NAVIGATION BAR ---
with st.sidebar:
    st.markdown(f"### 👤 <span style='color:#00f2ff'>{st.session_state.user_fullname if st.session_state.logged_in else 'GUEST MODE'}</span>", unsafe_allow_html=True)
    selected = option_menu("ChronosID Menu", ["Dashboard", "Analysis", "Dataset"], 
                           icons=["house", "camera", "database"], default_index=0,
                           styles={
                               "container": {"background": "transparent"},
                               "nav-link-selected": {"background-color": "#00f2ff", "color": "black"}
                           })
    if st.session_state.logged_in and st.button(" LOGOUT"):
        st.session_state.logged_in = False
        st.rerun()

# --- 7. PAGE LOGIC ---
if selected == "Dashboard":
    st.title("ChronosID Analytics - Hyper Vision")
    
    col1, col2 = st.columns([2, 1.2], gap="large")
    
    with col1:
        # 1. TYPEWRITER HEADER (Fixed Syntax)
        st.markdown("""
            <div class="typewriter-viewport">
                <h3 class="typewriter-text">Discover the Power Behind Every Face</h3>
            </div>
            <style>
            .typewriter-viewport {
                display: flex;
                justify-content: flex-start;
                align-items: center;
                height: 60px;
                margin-bottom: 20px;
            }
            .typewriter-text {
                color: #FF3131 !important;
                font-family: 'Courier New', Courier, monospace !important;
                font-size: 1.6rem !important;
                white-space: nowrap;
                overflow: hidden;
                border-right: 3px solid #FF3131 !important;
                width: 0;
                animation: 
                    typing 3.5s steps(35, end) forwards,
                    blink-caret 0.75s step-end infinite;
                text-shadow: 0 0 10px rgba(0, 242, 255, 0.7) !important;
            }
            @keyframes typing { from { width: 0 } to { width: 100% } }
            @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: #00f2ff } }
            </style>
        """, unsafe_allow_html=True)
        
        # Grid Layout for 3 Rows of 2 Cards
        row1 = st.columns(2)
        row2 = st.columns(2)
        row3 = st.columns(2)

        # ROW 1: CORE BIOMETRICS
        with row1[0]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#00f2ff; font-size:1.1rem;">Smart Age Analysis</b>
                    <p style="font-size:0.85rem; margin-top:5px;">AI-driven neural age estimation.</p>
                    <details style="cursor:pointer; font-size:0.8rem; color:#00f2ff;">
                        <summary>Read More</summary>
                        <p style="color:white; opacity:0.8; margin-top:5px;">
                        Utilizes deep neural networks to extract facial features and estimate age by analyzing skin texture and bone structure patterns.
                        </p>
                    </details>
                </div>
            """, unsafe_allow_html=True)
        with row1[1]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#00f2ff; font-size:1.1rem;">Accurate Gender Detection</b>
                    <p style="font-size:0.85rem; margin-top:5px;">Advanced gender identification.</p>
                    <details style="cursor:pointer; font-size:0.8rem; color:#00f2ff;">
                        <summary>Read More</summary>
                        <p style="color:white; opacity:0.8; margin-top:5px;">
                        Performs binary gender classification through high-dimensional facial vector analysis, ensuring high accuracy across diverse ethnicities.
                        </p>
                    </details>
                </div>
            """, unsafe_allow_html=True)

        # ROW 2: DATA & REPORTING
        with row2[0]:
            st.markdown("""
              <div class="metric-card">
                    <b style="color:#00f2ff; font-size:1.1rem;">MongoDB Cloud Storage</b>
                    <p style="font-size:0.85rem; margin-top:5px;">Secure MongoDB cloud integration.</p>
                    <details style="cursor:pointer; font-size:0.8rem; color:#00f2ff;">
                        <summary>Read More</summary>
                        <p style="color:white; opacity:0.8; margin-top:5px;">
                        Securely transmits and stores user telemetry and scan logs in a MongoDB Atlas cloud cluster for global data accessibility.
                        </p>
                    </details>
                </div>
            """, unsafe_allow_html=True)
        with row2[1]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#00f2ff; font-size:1.1rem;">PDF Report Generator</b>
                    <p style="font-size:0.85rem; margin-top:5px;">Standardized PDF reporting.</p>
                    <details style="cursor:pointer; font-size:0.8rem; color:#00f2ff;">
                        <summary>Read More</summary>
                        <p style="color:white; opacity:0.8; margin-top:5px;">
                        Generates standardized PDF biometric reports including subject IDs, timestamps, and cropped face profiles for professional use.
                        </p>
                    </details>
                </div>
            """, unsafe_allow_html=True)

        # ROW 3: INPUT & SCALABILITY
        with row3[0]:
            st.markdown("""
              <div class="metric-card">
                    <b style="color:#00f2ff; font-size:1.1rem;">Multiple Face Recognition</b>
                    <p style="font-size:0.85rem; margin-top:5px;">Parallel processing for group scans.</p>
                    <details style="cursor:pointer; font-size:0.8rem; color:#00f2ff;">
                        <summary>Read More</summary>
                        <p style="color:white; opacity:0.8; margin-top:5px;">
                        Capable of parallel processing multiple faces within a single frame, identifying age and gender for everyone detected simultaneously.
                        </p>
                    </details>
                </div>
            """, unsafe_allow_html=True)
        with row3[1]:
            st.markdown("""
                <div class="metric-card">
                    <b style="color:#00f2ff; font-size:1.1rem;">Live Camera & File Upload</b>
                    <p style="font-size:0.85rem; margin-top:5px;">Live and static input support.</p>
                    <details style="cursor:pointer; font-size:0.8rem; color:#00f2ff;">
                        <summary>Read More</summary>
                        <p style="color:white; opacity:0.8; margin-top:5px;">
                        A versatile input gateway supporting real-time live webcam streams as well as static file uploads for offline biometric analysis.
                        </p>
                    </details>
                </div>
            """, unsafe_allow_html=True)
        
        # 3D CUBE - THE CORE VISUAL CENTERPIECE
        st.markdown("""
            <div class="single-cube-container">
                <div class="cube">
                    <div class="face front"></div><div class="face back"></div>
                    <div class="face right"></div><div class="face left"></div>
                    <div class="face top"></div><div class="face bottom"></div>
                </div>
            </div>
            <p style='text-align:center; font-size: 0.9rem; color: #00f2ff; letter-spacing: 2px; text-shadow: 0 0 5px #00f2ff;'>Welcome to ChronosID...</p>
        """, unsafe_allow_html=True)

    with col2:

        if not st.session_state.logged_in:

            st.markdown('<div class="auth-card"> ACCESS PORTAL', unsafe_allow_html=True)

            t1, t2 = st.tabs(["LOGIN", "REGISTER"])

            with t1:

                u, p = st.text_input("Username"), st.text_input("Password", type="password")

                if st.button("AUTHENTICATE"):

                    if db is not None:

                        user = db.users.find_one({"username": u, "password": p})

                        if user:

                            st.session_state.logged_in, st.session_state.user_fullname = True, user["name"]

                            st.rerun()

                        else: st.error("Invalid Credentials")

                    else: st.error(" DATABASE OFFLINE")

            with t2:

                rn, ru, rp = st.text_input("Full Name"), st.text_input("New User"), st.text_input("New Pass", type="password")

                if st.button("CREATE PROFILE"):

                    if db is not None:

                        db.users.insert_one({"name": rn, "username": ru, "password": rp})

                        st.success("Authorized!")

                    else: st.error(" DATABASE OFFLINE")

            st.markdown('</div>', unsafe_allow_html=True)
elif selected == "Analysis":
    st.title(" CHRONOSID VISION ENGINE")
    if not st.session_state.logged_in:
        st.warning(" SECURE ACCESS REQUIRED.")
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
                st.error(" BIOMETRIC DATA NOT RECOGNIZED")
            else:
                display_img = input_image.copy()
                analysis_results = []
                
                st.markdown(f" SUBJECT TELEMETRY (Detected: {len(faces)})")
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
                        st.write(f"Gender: {g_lab}")
                        st.write(f"Range: {age_range} Yrs")
                        st.write(f"Confidence:{acc}%")

                st.markdown("---")
                st.markdown('<div class="scanner-container"><div class="scanner-line"></div>', unsafe_allow_html=True)
                st.image(display_img, channels="BGR", caption="Full Biometric Scan Overlay", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # PDF Button
                pdf_data = generate_pdf_report(st.session_state.user_fullname, analysis_results)
                st.download_button(label=" DOWNLOAD BIOMETRIC REPORT (PDF)", data=pdf_data, 
                                   file_name=f"ChronosID_Report_{datetime.now().strftime('%H%M%S')}.pdf", mime="application/pdf")
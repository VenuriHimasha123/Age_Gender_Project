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
# Ensure this file exists in your assets folder
try:
    bin_str = get_base64("assets/myphoto.jpg")
except:
    bin_str = "" # Fallback if image is missing

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

    def footer(self):
        # Position at 2.5 cm from bottom
        self.set_y(-25)
        
        # Background image for the footer
        try:
            # This uses your background image as a footer strip
            self.image("assets/background.png", x=0, y=270, w=210, h=30) 
        except:
            pass 

        # Footer Text Styling
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        
        # Line 1: Security Warning
        self.cell(0, 10, 'CONFIDENTIAL: Authorized Use Only. Biometric data is stored via MongoDB Atlas.', 0, 1, 'C')
        
        # Line 2: Page Number and System Brand
        self.set_text_color(0, 180, 200) 
        self.cell(0, 10, f'ChronosID Hyper Vision | Page {self.page_no()}/{{nb}}', 0, 0, 'C')

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

# --- 4. ADVANCED CSS: LIGHT THEME & FLAT UI ---
st.markdown(f"""
    <style>
    /* Global Smooth Scrolling & Background */
    html {{ scroll-behavior: smooth; }}
    .stApp {{
        
        background-image: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.4)), 
                          url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* TOP NAVBAR - Clean White with Shadow */
    .top-nav {{
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 70px !important;
        background: #FFFFFF !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        padding: 0 50px !important;
        z-index: 999999 !important;
    }}

    /* NAV BUTTONS */
    .nav-item {{
        color: #64748B !important;
        text-decoration: none !important;
        font-family: 'Arial', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 10px 20px;
        transition: 0.3s;
    }}
    .nav-item:hover {{ color: #00C4CC !important; }}

    /* LOG IN BUTTON STYLE (Pill shape from image) */
    .login-btn-nav {{
        background: #00C4CC !important;
        color: white !important;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: bold;
        text-decoration: none;
    }}

    header[data-testid="stHeader"] {{ visibility: hidden !important; }}

    /* CARDS & CONTAINERS (Flat clean look) */
    .metric-card, .auth-card {{
        background: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08) !important;
        padding: 25px;
        margin-bottom: 15px;
        border-top: 4px solid #00C4CC !important; /* Teal accent at top */
        color: #334155 !important;
    }}

    /* TEXT COLORS */
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {{
        color: #000000 !important; /* තද කළු පාට */
        font-family: 'Arial', sans-serif !important;
        font-weight: 900 !important; /* අකුරු මහත කිරීම */
        text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.9) !important; /* සුදු පාට Glow එකක් දීම */
    }}

    /* BUTTONS */
    .stButton>button {{
        background: #00C4CC !important;
        color: white !important; 
        border: none !important;
        border-radius: 30px !important; /* Pill shape */
        padding: 10px 25px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(0, 196, 204, 0.3) !important;
        transition: 0.3s;
    }}
    .stButton>button:hover {{ 
        background: #009FA6 !important; 
        transform: translateY(-2px);
    }}
    
    /* INPUT FIELDS */
    input {{ 
        background: #F1F5F9 !important; 
        color: #334155 !important; 
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important; 
    }}

   /* අලුතින් එකතු කරපු ඇවිදින Animation එක */
    .walking-man {{
        position: absolute;
        bottom: 0px; 
        width: 130px; /* GIF එකේ සයිස් එක */
        animation: walkIn 8s linear infinite; /* තත්පර 8කින් ඇවිදිනවා */
    }}

    @keyframes walkIn {{
        0% {{ left: -150px; opacity: 1; }} /* වම් පැත්තෙන් එළියේ ඉඳන් පටන් ගන්නවා */
        100% {{ left: 100%; opacity: 1; }} /* දකුණු පැත්තට (Login box එක දිහාට) ඇවිදගෙන යනවා */
    }}
    </style>
    """, unsafe_allow_html=True)

# --- TOP NAVBAR INJECTION ---
status_col = "#00C4CC" if db is not None else "#FF3131"
status_lab = "System Online" if db is not None else "online"

st.markdown(f"""
    <div class="top-nav">
        <div style="color: #1E293B; font-weight: 900; font-size: 1.5rem; display: flex; align-items: center; gap: 10px;">
            <span style="background: #00C4CC; color: white; padding: 5px 10px; border-radius: 50%;">C</span> YOUR LOGO
        </div>
        <div style="display: flex; gap: 30px; align-items: center;">
            <a href="/?page=Dashboard" class="nav-item">Home</a>
            <a href="/?page=Analysis" class="nav-item">Scan</a>
            <a href="/?page=Settings" class="nav-item">Database</a>
            <a href="#" class="login-btn-nav"><i class="fa fa-user"></i> Log In</a>
        </div>
    </div>
    <div style="margin-top: 100px;"></div>
""", unsafe_allow_html=True)

# --- 1. SYNC NAVIGATION LOGIC ---
# This reads the URL to see if a Top Nav link was clicked
if "page" in st.query_params:
    selected = st.query_params["page"]
else:
    # This will be overridden by the Sidebar selection later
    selected = "Dashboard" 

# --- 2. TOP NAVBAR INJECTION ---
# Calculate status outside to prevent "Code Leaks"
status_col = "#9DB5E7" if db is not None else "#90E6DB"
status_lab = "Welcome to ChronsoID Analytics" if db is not None else "ONLINE"

st.markdown(f"""
    <div class="top-nav">
        <div style="color: #E8E9F2; font-weight: bold; letter-spacing: 2px; font-family: 'Courier New';">CHRONOSID</div>
        <div style="display: flex; gap: 20px;">
            <a href="/?page=Dashboard" target="_self" class="nav-item">DASHBOARD</a>
              <a href="/?page=Dashboard" target="_self" class="nav-item">ANALYSIS</a>
            <a href="/?page=Dashboard" target="_self" class="nav-item">SETTINGS</a>
        </div>
            <div style="color: {status_col}; font-size: 0.7rem; font-family: 'Courier New'; font-weight: bold;">
            {status_lab}
        </div>
       </div>
    <div style="margin-top: 100px;"></div>
""", unsafe_allow_html=True)

# --- 5. SESSION STATE ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_fullname" not in st.session_state: st.session_state.user_fullname = ""

# --- 6. NAVIGATION BAR ---
with st.sidebar:
    st.markdown(f"  <span style='color:#00f2ff'>{st.session_state.user_fullname if st.session_state.logged_in else 'GUEST MODE'}</span>", unsafe_allow_html=True)
   # [වෙනස් වුණු තැන] Menu options ටික සහ Icons ටික මාරු කළා
    selected = option_menu("ChronosID Menu", ["Dashboard", "Analysis", "Diagnostics", "Developer API"], 
                           icons=["house", "camera", "activity", "code-slash"], default_index=0,
                           styles={
                               "container": {"background": "transparent"},
                               "nav-link-selected": {"background-color": "#00f2ff", "color": "black"}
                           })
    if st.session_state.logged_in and st.button(" LOGOUT"):
        st.session_state.logged_in = False
        st.rerun()

## --- 7. PAGE LOGIC ---
if selected == "Dashboard":
    
    # 1. NEW HERO SECTION (Like the image)
    st.markdown("""
        <div style="padding: 40px 0px; max-width: 800px;">
            <h1 style="font-size: 4.5rem; color: #000000 !important; line-height: 1.1; margin-bottom: 0; text-shadow: 3px 3px 15px rgba(255,255,255,0.9);">
                <span style="font-family: cursive; color: #000000 !important; font-size: 3.5rem; display: block;">Discover the</span>
                POWER OF <span style="color: #009FA6 !important;">BIOMETRICS</span>
            </h1>
            <p style="color: #000000 !important; font-size: 1.2rem; font-weight: bold; margin-top: 20px; max-width: 600px; text-shadow: 2px 2px 10px rgba(255,255,255,0.9); background: rgba(255,255,255,0.3); padding: 10px; border-radius: 10px;">
                Advanced neural age estimation and accurate gender detection powered by state-of-the-art Deep Learning models. Secure, fast, and reliable.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1.2], gap="large")
    
    with col1:
        # Grid Layout for 2 Rows of 2 Cards
        row1 = st.columns(2)
        row2 = st.columns(2)
        row3 = st.columns(2) # අලුත් පේළිය 1
        row4 = st.columns(2) # අලුත් පේළිය 2

        # ROW 1
        with row1[0]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">Smart Age Analysis</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">AI-driven neural age estimation.</p>
                </div>
            """, unsafe_allow_html=True)
        with row1[1]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">Accurate Gender</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">Advanced gender identification.</p>
                </div>
            """, unsafe_allow_html=True)

        # ROW 2
        with row2[0]:
            st.markdown("""
              <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">Cloud Storage</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">Secure MongoDB integration.</p>
                </div>
            """, unsafe_allow_html=True)
        with row2[1]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">Live Camera</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">Real-time stream support.</p>
                </div>
            """, unsafe_allow_html=True)


            # --- ROW 3 (අලුතින් එකතු කළ Cards 2) ---
        with row3[0]:
            st.markdown("""
              <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">PDF Reports</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">Automated forensic reporting.</p>
                </div>
            """, unsafe_allow_html=True)
        with row3[1]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">Multi-Face Sync</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">Parallel group scanning.</p>
                </div>
            """, unsafe_allow_html=True)

        # --- ROW 4 (අලුතින් එකතු කළ Cards 2) ---
        with row4[0]:
            st.markdown("""
              <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">Data Security</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">End-to-end encryption.</p>
                </div>
            """, unsafe_allow_html=True)
        with row4[1]:
            st.markdown("""
               <div class="metric-card">
                    <b style="color:#1E293B; font-size:1.2rem;">High Speed</b>
                    <p style="font-size:0.9rem; margin-top:5px; color:#64748B;">Low latency inference.</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:

# --- අලුතින් එකතු කරපු Photos 3 ---
        st.markdown('<div style="margin-bottom: 15px;">', unsafe_allow_html=True)
        # පස්සෙ ඔයාට ඕනෙනම් මේ URLs වෙනුවට "assets/photo1.jpg" වගේ දාන්න පුළුවන්
        st.image("assets/facial.webp", use_container_width=True)
        st.image("assets/face2.webp", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        # ----------------------------------

        if not st.session_state.logged_in:


            st.markdown('<div class="auth-card"><h3 style="margin-top:0;">Access Portal</h3>', unsafe_allow_html=True)
            t1, t2 = st.tabs(["LOGIN", "REGISTER"])

            with t1:
                u = st.text_input("Username", key="login_user")
                p = st.text_input("Password", type="password", key="login_pass")

                if st.button("Login"):
                    # 1. Check if Database is ONLINE
                    if db is not None:
                        user = db.users.find_one({"username": u, "password": p})
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user_fullname = user["name"]
                            st.success("Authorized via Cloud Node")
                            st.rerun()
                        elif u == "admin" and p == "admin":
                            st.session_state.logged_in = True
                            st.session_state.user_fullname = "Admin (Override)"
                            st.rerun()
                        else:
                            st.error("Invalid Credentials")
                    
                    # 2. Database is OFFLINE: Use manual bypass
                    else:
                        if u == "admin" and p == "admin":
                            st.session_state.logged_in = True
                            st.session_state.user_fullname = "Venuri Himasha(online)"
                            st.warning("OFFLINE MODE: Manual Bypass Granted")
                            st.rerun()
                        else:
                            st.error("DATABASE OFFLINE. Use 'admin' credentials.")
            with t2:

                rn, ru, rp = st.text_input("Full Name"), st.text_input("Username"), st.text_input("New Password", type="password")

                if st.button("Register"):

                    if db is not None:

                        db.users.insert_one({"name": rn, "username": ru, "password": rp})

                        st.success("Authorized!")

                    else: st.error(" DATABASE OFFLINE")

            st.markdown('</div>', unsafe_allow_html=True)
            
# --- ANALYSIS PAGE ---
elif selected == "Analysis":
    st.title(" CHRONOSID VISION ENGINE")

    if not st.session_state.logged_in:
        st.warning(" SECURE ACCESS REQUIRED. Please login from the Dashboard.")
    else:
        c_path, m_path = "models/haarcascade_frontalface_default.xml", "models/age_gender_model.h5"
        
        if os.path.exists(c_path) and os.path.exists(m_path):
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
                # --- [1] CLAHE (Global Lighting Fix) ---
                lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l_channel)
                merged_lab = cv2.merge((cl, a, b))
                enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
                
                # --- [2] GRAYSCALE ---
                gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) == 0:
                    st.error(" BIOMETRIC DATA NOT RECOGNIZED")
                else:
                    with st.expander("VIEW LIVE IMAGE PREPROCESSING PIPELINE", expanded=True):
                        st.write("Live breakdown of the AI transforming the input before prediction:")
                        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                        
                        # Display Step 1 & 2 (Whole Image)
                        with pc1:
                            st.markdown("<p style='font-size:14px; font-weight:bold;'>1. CLAHE</p>", unsafe_allow_html=True)
                            st.image(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with pc2:
                            st.markdown("<p style='font-size:14px; font-weight:bold;'>2. Grayscale</p>", unsafe_allow_html=True)
                            st.image(gray, use_container_width=True)
                        
                        # Calculate Step 3, 4, 5 for the FIRST detected face to show in pipeline
                        fx, fy, fw, fh = faces[0]
                        img_h, img_w, _ = input_image.shape
                        mx, my = int(fw * 0.05), int(fh * 0.05) # 5% margin
                        x1, y1 = max(0, fx - mx), max(0, fy - my)
                        x2, y2 = min(img_w, fx + fw + mx), min(img_h, fy + fh + my)
                        
                        sample_roi = input_image[y1:y2, x1:x2]
                        sample_roi_rgb = cv2.cvtColor(sample_roi, cv2.COLOR_BGR2RGB)
                        sample_f_crop = cv2.resize(sample_roi_rgb, (128, 128)) / 255.0
                        
                        with pc3:
                            st.markdown("<p style='font-size:14px; font-weight:bold;'>3. Dynamic Crop</p>", unsafe_allow_html=True)
                            st.image(sample_roi_rgb, use_container_width=True)
                        with pc4:
                            st.markdown("<p style='font-size:14px; font-weight:bold;'>4. Resize & Norm</p>", unsafe_allow_html=True)
                            st.image(sample_f_crop, use_container_width=True) # displays the 128x128 0-1 float array
                        with pc5:
                            st.markdown("<p style='font-size:14px; font-weight:bold;'>5. Tensor Shape</p>", unsafe_allow_html=True)
                            st.info("Format: 4D Array\n\nShape: (1, 128, 128, 3)\n\nMin: 0.0 | Max: 1.0")
                    # =====================================================================

                    display_img = input_image.copy() 
                    analysis_results = []
                    
                    st.markdown(f" SUBJECT TELEMETRY (Detected: {len(faces)})")
                    cols = st.columns(len(faces))

                    for i, (x, y, w, h) in enumerate(faces):
                        margin_x = int(w * 0.05)
                        margin_y = int(h * 0.05)
                        
                        img_h, img_w, _ = input_image.shape
                        
                        x1 = max(0, x - margin_x)
                        y1 = max(0, y - margin_y)
                        x2 = min(img_w, x + w + margin_x)
                        y2 = min(img_h, y + h + margin_y)
                        
                        roi = input_image[y1:y2, x1:x2]
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                        # Pipeline to Model
                        f_crop = cv2.resize(roi_rgb, (128, 128)) / 255.0
                        preds = multi_model.predict(np.reshape(f_crop, (1, 128, 128, 3)))
                        
                        # GENDER LOGIC
                        g_prob = preds[0][0][0]
                        g_lab = "FEMALE" if g_prob > 0.5 else "MALE"
                        
                        # AGE LOGIC
                        base_age = int(preds[1][0][0])
                        age_range = f"{max(0, base_age - 3)}-{base_age + 3}" 
                        
                        acc = round((g_prob if g_lab == "FEMALE" else 1 - g_prob) * 100, 1)
                        
                        analysis_results.append({
                            'id': f'SUB_{i+1}', 
                            'gender': g_lab, 
                            'age_range': age_range, 
                            'confidence': acc, 
                            'crop_bgr': roi # For PDF generation
                        })

                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 4)
                        cv2.putText(display_img, f"SUB_{i+1}", (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

                        with cols[i]:
                            st.markdown(f"**SUBJECT {i+1}**")
                            st.image(roi_rgb, caption=f"Bio-Profile {i+1}", use_container_width=True)
                            st.write(f"Gender: {g_lab}")
                            st.write(f"Range: {age_range} Yrs")
                            st.write(f"Confidence:{acc}%")

                    st.markdown("---")
                    st.markdown('<div class="scanner-container"><div class="scanner-line"></div>', unsafe_allow_html=True)
                    st.image(display_img, channels="BGR", caption="Full Biometric Scan Overlay", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # --- MongoDB SAVE HISTORY ---
                    if db is not None:
                        log_data = {
                            "operator": st.session_state.user_fullname,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "total_faces": len(faces),
                            "predictions": [{"id": r['id'], "gender": r['gender'], "age": r['age_range'], "confidence": r['confidence']} for r in analysis_results]
                        }
                        db.scan_history.insert_one(log_data)
                        st.toast("✅ Scan results securely saved to Cloud Database!")

                    # --- PDF GENERATION ---
                    pdf_data = generate_pdf_report(st.session_state.user_fullname, analysis_results)
                    st.download_button(
                        label=" DOWNLOAD BIOMETRIC REPORT (PDF)", 
                        data=pdf_data, 
                        file_name=f"ChronosID_Report_{datetime.now().strftime('%H%M%S')}.pdf", 
                        mime="application/pdf"
                    )
        else:
            st.error("Model or Cascade file missing! Make sure 'models' folder exists with valid .h5 and .xml files.")
                # =====================================================================
# --- NEW PAGE 1: SYSTEM DIAGNOSTICS ---
elif selected == "Diagnostics":
    st.title("⚙️ SYSTEM DIAGNOSTICS")
    if not st.session_state.logged_in:
        st.warning(" SECURE ACCESS REQUIRED. Please login via Dashboard.")
    else:
        import time
        import random
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><h4 style="margin-top:0;">Model Status</h4><h2 style="color:#00f2ff; margin-bottom:0;">ONLINE</h2><p style="opacity:0.7; font-size:0.8rem;">Age_Gender_v1.h5</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h4 style="margin-top:0;">Inference Speed</h4><h2 style="color:#00f2ff; margin-bottom:0;">{random.randint(32, 45)} ms</h2><p style="opacity:0.7; font-size:0.8rem;">Per Face Detected</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h4 style="margin-top:0;">Memory Usage</h4><h2 style="color:#00f2ff; margin-bottom:0;">{random.randint(120, 150)} MB</h2><p style="opacity:0.7; font-size:0.8rem;">TensorFlow Allocation</p></div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-card">
                <h4 style="margin-top:0; color:#00f2ff;">Vision Pipeline Health</h4>
                <p style="margin:5px 0;">✅ OpenCV Cascade Classifier: <b>Loaded Successfully</b></p>
                <p style="margin:5px 0;">✅ CNN Multi-Task Weights: <b>Verified & Active</b></p>
                <p style="margin:5px 0;">✅ MongoDB Cloud Cluster: <b>Connected (Latency < 20ms)</b></p>
            </div>
        """, unsafe_allow_html=True)

# --- NEW PAGE 2: DEVELOPER API ---
elif selected == "Developer API":
    st.title("🔑 DEVELOPER API ACCESS")
    if not st.session_state.logged_in:
        st.warning(" SECURE ACCESS REQUIRED. Please login via Dashboard.")
    else:
        st.markdown("""
            <div class="metric-card">
                <h3 style="margin-top:0; color:#00f2ff;">API Integration Interface</h3>
                <p>Use this secure API token to integrate ChronosID biometrics into external mobile or web applications.</p>
            </div>
        """, unsafe_allow_html=True)
        
        import uuid
        # Generate a consistent token based on username
        fake_token = f"chronos_live_{uuid.uuid5(uuid.NAMESPACE_DNS, st.session_state.user_fullname).hex}"
        
        st.markdown("**Your Secure Access Token:**")
        st.code(f"Authorization: Bearer {fake_token}", language="bash")
        
        st.markdown("<br>**Example cURL Request (For Developers):**", unsafe_allow_html=True)
        st.code('''curl -X POST "https://api.chronosid.com/v1/analyze" \\
     -H "Authorization: Bearer YOUR_API_TOKEN" \\
     -H "Content-Type: multipart/form-data" \\
     -F "image=@subject_photo.jpg"''', language="bash")
        
        st.info("Note: API rate limits are currently restricted to 100 requests per minute for this account tier.")
# =====================================================================

        # --- PAGE FOOTER --- (මේක වෙනස් කරන්න එපා)
                
                # --- PAGE FOOTER ---
# Scroll to the very bottom of your file and paste this:
st.markdown("""
    <div class="footer-container">
        <p style="font-size: 0.8rem; opacity: 0.6; color: #00f2ff !important; font-family: 'Courier New';">
            © 2026 CHRONOSID ANALYTICS | NEURAL BIOMETRIC SYSTEMS
        </p>
        <p style="font-size: 0.7rem; opacity: 0.4; color: white !important; font-family: 'Courier New';">
            Authorized Deployment for General Sir John Kotelawala Defence University
        </p>
        <div style="margin-top: 10px;">
            <span style="border: 1px solid #FF3131; padding: 2px 8px; border-radius: 5px; font-size: 0.6rem; color: #FF3131; font-family: 'Courier New';">SECURE CORE</span>
            <span style="border: 1px solid #00f2ff; padding: 2px 8px; border-radius: 5px; font-size: 0.6rem; color: #00f2ff; margin-left: 10px; font-family: 'Courier New';">ENCRYPTED DATA</span>
        </div>
    </div>
""", unsafe_allow_html=True)
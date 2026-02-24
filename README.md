<img width="1875" height="858" alt="imgg" src="https://github.com/user-attachments/assets/7c19be35-7645-4203-9aea-e78c326f942e" />



# ChronosID Analytics: Biometric Intelligence Engine
ChronosID Analytics is a high-performance neural vision platform designed for real-time age and gender estimation. Built with a futuristic Cyber-Neon aesthetic, it combines deep learning with cloud data persistence and advanced 3D web rendering.

#  Core Features
Neural Vision Scanner: Real-time biometric analysis using OpenCV and TensorFlow/Keras.

Hyper-Glassmorphism UI: High-end frosted glass interface with saturation filters and smooth global scrolling.

Dual-Mode Biometrics: Supports both Live Camera Scanner and Static Image Uploads.

Cloud Data Persistence: Integrated with MongoDB Atlas for secure user management and telemetry storage.

##  Tech Stack
Category	Technology
Frontend	Streamlit, Custom CSS3 (Preserve-3D)
Deep Learning	TensorFlow, Keras, Haar Cascades
Backend	Python, MongoDB Atlas (NoSQL)
Computer Vision	OpenCV (CV2), NumPy
Documentation	FPDF, UUID Tracking


##  Installation & Setup
Clone the Repository

Bash
git clone https://github.com/VenuriHimasha123/Age_Gender_Project.git
cd ChronosID-Analytics
Install Dependencies

Bash
pip install streamlit streamlit-option-menu "pymongo[srv]" certifi tensorflow opencv-python fpdf
Configure Database

Update the MONGO_URI in app.py with your MongoDB Atlas connection string.

Ensure your IP address is whitelisted in the MongoDB Atlas dashboard.

Launch the Engine

Bash
python -m streamlit run scripts/app.py


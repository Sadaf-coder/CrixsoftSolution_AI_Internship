# 👤 Face Detection using OpenCV in Python
### Crixsoft Solution Internship — AI Domain

---

## 📌 Project Overview
This project is a real-time **Face Detection System** built using Python and OpenCV.
It detects human faces through webcam or uploaded images and highlights them with bounding boxes.

---

## 🔍 Detection Details
| Feature | Description |
|---------|-------------|
| 👤 Face | Detected with green bounding box |
| 👁️ Eyes | Detected with blue bounding box |
| 📷 Mode | Camera or Image upload |

---

## 🛠️ Technologies Used
- **Python 3.14**
- **OpenCV** — Face & Eye Detection using Haar Cascades
- **NumPy** — Numerical Processing

---

## 🔍 How It Works
1. User selects Camera or Image mode
2. Haar Cascade detects faces in frame
3. Eyes are detected in upper 60% of face
4. Green box drawn around face
5. Blue box drawn around eyes
6. Total face count displayed on screen

---

## 📁 Project Structure
```
Face Detection using OpenCV
│
├── face_detection.py   # Main project file
├── requirements.txt    # Required libraries
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/CrixsoftSolution_AI_FaceDetection.git
```

### 2. Create virtual environment
```bash
python -m venv face_env
source face_env/Scripts/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python face_detection.py
```

### 5. Choose mode
```
1 — Use Camera
2 — Upload Image
```

---

## 🎯 Features
- ✅ Real-time face detection via camera
- ✅ Image upload face detection
- ✅ Eye detection
- ✅ Face count display
- ✅ Fullscreen camera mode
- ✅ Green bounding box around face
- ✅ Blue bounding box around eyes

---

## 👩‍💻 Developed By
**Sadaf** — AI Intern at Crixsoft Solution

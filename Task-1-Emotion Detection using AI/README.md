# 🎭 Emotion Detection System

## 📌 Project Overview
This project is a real-time **Emotion Detection System** built using Python and OpenCV.
It detects human facial expressions through a webcam and identifies emotions like Happy, Neutral, Sleepy and Winking.

---

## 😊 Detected Emotions
| Emoji | Emotion | Trigger |
|-------|---------|---------|
| 😊 | Happy | Smile detected for 5 frames |
| 😐 | Neutral | Normal face |
| 😴 | Sleepy | Both eyes closed |
| 😉 | Winking | One eye closed |

---

## 🛠️ Technologies Used
- **Python 3.14**
- **OpenCV** — Face, Eye, Smile Detection using Haar Cascades
- **NumPy** — Numerical Processing

---

## 🔍 How It Works
1. Captures real-time video from webcam
2. Detects face using Haar Cascade
3. Detects eyes in upper 60% of face
4. Detects smile in lower face region
5. Smile counter counts 5 frames then shows Happy
6. Tracks emotion history and displays dominant emotion

---

## 📁 Project Structure

Task-1-Emotion Detection using AI/
│
├── emotion_detection.py   # Main project file
├── requirements.txt       # Required libraries
├── README.md              # Project documentation
└── emotion_env            # Virtual environment

---

## 🎯 Features
- ✅ Real-time face detection
- ✅ Live emotion recognition
- ✅ Emotion history tracking
- ✅ Dominant emotion display
- ✅ Smile counter display
- ✅ Eye zone highlighting
- ✅ Fullscreen mode support
- ✅ Stats box on screen

---

## 👩‍💻 Developed By
**Sadaf** — AI Intern at Crixsoft Solution



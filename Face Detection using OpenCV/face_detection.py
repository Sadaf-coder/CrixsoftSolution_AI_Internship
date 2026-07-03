import cv2
import sys
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

def detect_faces_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image is not loaded!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey),
                         (ex+ew, ey+eh), (255, 0, 0), 2)

        cv2.putText(img, f"Face Detected!", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Header
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(img, "CRIXSOFT INTERNSHIP - Face Detection",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    cv2.putText(img, f"Faces Found: {len(faces)}",
                (img.shape[1]-200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.namedWindow("Face Detection - Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_camera():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Face Detection - Camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face Detection - Camera",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    print("Camera mode... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            eye_zone = face_gray[0:int(h*0.6), 0:w]
            eye_zone_color = face_color[0:int(h*0.6), 0:w]

            eyes = eye_cascade.detectMultiScale(
                eye_zone,
                scaleFactor=1.1,
                minNeighbors=15,
                minSize=(30, 30),
                maxSize=(80, 80)
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(eye_zone_color,
                             (ex, ey), (ex+ew, ey+eh),
                             (255, 0, 0), 2)

            # Badge
            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), -1)
            cv2.putText(frame, f"Face", (x+5, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)

        # Header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "CRIXSOFT INTERNSHIP - Face Detection",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("Face Detection - Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---- MAIN ----
print("\n============================")
print(" CRIXSOFT - Face Detection ")
print("============================")
print("1 — Use Camera")
print("2 — Upload Image")
choice = input("Choose (1 or 2): ")

if choice == "1":
    detect_faces_camera()
elif choice == "2":
    path = input("Write Image Path: ")
    detect_faces_image(path)
else:
    print("Invalid choice!")
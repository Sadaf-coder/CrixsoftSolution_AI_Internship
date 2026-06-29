import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Emotion Detection - Crixsoft", cv2.WINDOW_NORMAL)  # ← ADD 1
print("Emotion Detection Started... Press 'q' to quit")

emotion_history = []
smile_counter = 0
SMILE_THRESHOLD = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for (x, y, w, h) in faces:
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

        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=25,
            minSize=(30, 30)
        )

       # Smile counter logic - aankhein band ho to turant reset
        if len(eyes) == 0:
            smile_counter = 0  # turant reset
        elif len(smiles) > 0:
            smile_counter += 1
        else:
            smile_counter = max(0, smile_counter - 1)   

        smile_detected = smile_counter >= SMILE_THRESHOLD

        if smile_detected:
            emotion = "Happy"
            emoji = ":)"
            color = (0, 255, 0)
        elif len(eyes) == 0:
            emotion = "Sleepy"
            emoji = "-_-"
            color = (0, 165, 255)
        elif len(eyes) == 1:
            emotion = "Winking"
            emoji = ";)"
            color = (255, 255, 0)
        else:
            emotion = "Neutral"
            emoji = ":|"
            color = (200, 200, 200)

        emotion_history.append(emotion)
        if len(emotion_history) > 30:
            emotion_history.pop(0)
        most_common = max(set(emotion_history),
                         key=emotion_history.count)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-45), (x+w, y), color, -1)
        cv2.putText(frame, f"{emoji} {emotion}",
                    (x+8, y-12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 0, 0), 2)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(eye_zone_color,
                         (ex, ey), (ex+ew, ey+eh),
                         (255, 255, 0), 1)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_color,
                         (sx, sy), (sx+sw, sy+sh),
                         (0, 255, 0), 1)

        stats_x = frame.shape[1] - 220
        cv2.rectangle(frame, (stats_x, 90),
                      (frame.shape[1]-10, 210),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (stats_x, 90),
                      (frame.shape[1]-10, 210),
                      color, 1)
        cv2.putText(frame, "STATS", (stats_x+10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"Current:  {emotion}",
                    (stats_x+10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Dominant: {most_common}",
                    (stats_x+10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Eyes: {len(eyes)}",
                    (stats_x+10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Smile: {smile_counter}/{SMILE_THRESHOLD}",
                    (stats_x+10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    cv2.putText(frame, "CRIXSOFT INTERNSHIP",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 255), 2)
    cv2.putText(frame, "Emotion Detection System",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Faces: {len(faces)}",
                (frame.shape[1]-200, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1)

    cv2.imshow("Emotion Detection - Crixsoft", frame)
    cv2.setWindowProperty("Emotion Detection - Crixsoft",  # ← ADD 2
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
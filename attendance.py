import cv2
import sqlite3
from datetime import datetime

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load label mapping
labels = {}
with open("labels.txt") as f:
    for line in f:
        k, v = line.strip().split(",")
        labels[int(k)] = v

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open DB connection once
conn = sqlite3.connect("attendance.db")
cur = conn.cursor()

# Memory to avoid duplicates in same session
marked_today = set()

# Start camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 60:
            name = labels[id_]
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            if name not in marked_today:
                cur.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
                data = cur.fetchall()
                if len(data) == 0:
                    cur.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
                    conn.commit()
                marked_today.add(name)

            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close DB and camera
conn.close()
cam.release()
cv2.destroyAllWindows()

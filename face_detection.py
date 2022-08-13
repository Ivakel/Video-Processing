import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_faces(faces):
    count = 0
    for _ in faces:
        count += 1
    return count


while True:

    ret, frame = cap.read()
    rows, cols, channels = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    frame = cv2.rectangle(frame, (30, 30), (100, 60), (0, 0, 0), 3)

    num_faces = "faces: " + str(get_faces(faces))

    font = cv2.FONT_HERSHEY_SIMPLEX

    frame = cv2.putText(frame, num_faces, (30, rows-30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 3)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
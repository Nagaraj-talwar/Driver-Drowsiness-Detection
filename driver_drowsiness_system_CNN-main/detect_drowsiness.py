import cv2
import numpy as np
import serial
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread

# Function to start the alarm sound
def start_alarm(sound):
    playsound('data/alarm.mp3')

# Classes for eye state classification
classes = ['Closed', 'Open']

# Load pre-trained models and cascades
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
# Initialize video capture from webcam
cap = cv2.VideoCapture(0)
model = load_model("drowiness_new7.h5")



# Initialize variables
count = 0
alarm_on = False
alarm_sound = "data/alarm.mp3"
status1 = ''  # Initialize with 'Open' assuming eyes start open
status2 = ''

# Initialize serial communication with Arduino
try:
    arduino = serial.Serial('COM4', 9600, timeout=1)  # Adjust the port as needed for macOS
    time.sleep(2)  # Allow time for Arduino to initialize
    print("Arduino connected successfully.")
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")
    arduino = None

while True:
    _, frame = cap.read()
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect left eye in the face region
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye=right_eye_cascade.detectMultiScale(roi_gray)
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1=np.argmax(pred1)
            # print(f"Left eye prediction: {pred1}")  # Debugging line
            # if len(pred1) > 0 and pred1.shape[1] == 2:
            #     status1 = classes[np.argmax(pred1)]
            # else:
            #     status1 = 'Unknown'  # Handle case where prediction fails
            break
        
        # Detect right eye in the face region
        #right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2=np.argmax(pred2)
            # print(f"Right eye prediction: {pred2}")  # Debugging line
            # if len(pred2) > 0 and pred2.shape[1] == 2:
            #     status2 = classes[np.argmax(pred2)]
            # else:
            #     status2 = 'Unknown'  # Handle case where prediction fails
            break
        
        # Determine if eyes are closed
        if status1 == 2 and status2 == 2:
            count += 1
            cv2.putText(frame, f"Eyes Closed, Frame count: {count}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            if count >= 10:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
                    if arduino:
                        arduino.write(b'1')  # Send signal to Arduino to activate vibrator
        else:
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False
            if arduino:
                arduino.write(b'0')  # Send signal to Arduino to deactivate vibrator
        
    cv2.imshow("Drowsiness Detector", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close serial connection with Arduino
if arduino:
    arduino.close()

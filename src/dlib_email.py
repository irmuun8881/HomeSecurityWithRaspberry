import cv2
import numpy as np
import dlib
import pickle
import smtplib
from email.message import EmailMessage
import imghdr
import os
from dotenv import load_dotenv
import datetime
import time

# Load environment variables
load_dotenv()

# Dlib's face detection and recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/dlib_face_recognition_resnet_model_v1.dat')

# Load known face encodings and names from the pickle file
with open('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Initialize video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Email credentials
sender_email = os.getenv('SENDER_EMAIL')
sender_password = os.getenv('SENDER_PASSWORD')
receiver_email = os.getenv('RECEIVER_EMAIL')

# Function to send email with an attachment
def send_email_with_attachment(subject, body, receiver_email, sender_email, sender_password, attachment):
    message = EmailMessage()
    message.set_content(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email

    # Add attachment
    if attachment:
        with open(attachment, 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = f.name
        message.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(message)

# Initialize counters and timers
face_detections_count = 0
known_faces_count = 0
unknown_faces_count = 0
unknown_faces_temp_count = 0
start_time = time.time()
unknown_face_start_time = None
total_frames_processed = 0
program_duration = 20  # Duration for which the program should run, in seconds
email_cooldown = 60  # Email cooldown in seconds (set to 60 seconds)
last_email_time = 0

# Calculate the end time based on the duration
end_time = start_time + program_duration

while time.time() < end_time:
    ret, frame = video_capture.read()
    if not ret:
        break

    total_frames_processed += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)
    
    if faces:
        face_detections_count += len(faces)

    face_encodings = []
    for face in faces:
        shape = shape_predictor(rgb_frame, face)
        face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))
        face_encodings.append(face_encoding)

        (top, right, bottom, left) = (face.top(), face.right(), face.bottom(), face.left())
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    for face_encoding in face_encodings:
        distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        match = np.any(distances <= 0.4)
        name = "Unknown"
        if match:
            first_match_index = np.argmin(distances)
            name = known_face_names[first_match_index]
            known_faces_count += 1
            print(f"Detected known person: {name}")
        else:
            if unknown_face_start_time is None:
                unknown_face_start_time = time.time()
            unknown_faces_temp_count += 1
            
            if time.time() - unknown_face_start_time > 5:
                if unknown_faces_temp_count >= 4:
                    img_name = f"unknown_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(img_name, frame)
                    send_email_with_attachment("Unrecognized Person Detected",
                                               "Several unknown people were detected.",
                                               receiver_email, sender_email, sender_password,
                                               img_name)
                    last_email_time = time.time()
                unknown_face_start_time = None
                unknown_faces_temp_count = 0
            unknown_faces_count += 1
            print("Detected unknown person.")

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Calculations for duration and speed
total_duration = time.time() - start_time
detection_speed = total_frames_processed / total_duration if total_duration > 0 else 0
faces_per_frame = face_detections_count / total_frames_processed if total_frames_processed > 0 else 0

# Print the results
print(f"Total known face detections: {known_faces_count}")
print(f"Total unknown face detections: {unknown_faces_count}")
print(f"Total face detections: {face_detections_count}")
print(f"Total processing time: {total_duration:.2f} seconds")
print(f"Detection speed: {detection_speed:.2f} frames per second")
print(f"Average faces detected per frame: {faces_per_frame:.2f}")
print(f"Total frames processed: {total_frames_processed}")

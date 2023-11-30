import cv2
import numpy as np
import dlib
import pickle
import threading
import time
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
import imghdr
import datetime

load_dotenv()

# Load pre-existing encodings
with open("/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/known_faces.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Dlib's face detection and recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/dlib_face_recognition_resnet_model_v1.dat')

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

# Shared state for threads
class SharedState:
    def __init__(self):
        self.last_email_time = 0
        self.unknown_faces_detected = 0
        self.unknown_face_start_time = None
        self.known_faces_count = 0
        self.unknown_faces_count = 0
        self.lock = threading.Lock()

shared_state = SharedState()

# Function to process each frame
def process_frame(frame, known_face_encodings, known_face_names):
    with shared_state.lock:
        email_cooldown = 15  # Email cooldown in seconds
        unknown_detection_window = 5  # Time window to detect unknown faces
        unknown_detection_count = 4  # Number of unknown faces to detect within the window
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_frame)
        for face in faces:
            shape = shape_predictor(rgb_frame, face)
            face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))

            # Compare face encodings
            distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
            best_match_index = np.argmin(distances)
            name = "Unknown"
            if distances[best_match_index] < 0.4:  # Threshold for recognizing a face
                name = known_face_names[best_match_index]
                shared_state.known_faces_count += 1
            else:
                shared_state.unknown_faces_count += 1
                if shared_state.unknown_face_start_time is None:
                    shared_state.unknown_face_start_time = time.time()
                    shared_state.unknown_faces_detected = 1
                else:
                    shared_state.unknown_faces_detected += 1
                    elapsed_time = time.time() - shared_state.unknown_face_start_time
                    if elapsed_time > unknown_detection_window:
                        if shared_state.unknown_faces_detected >= unknown_detection_count:
                            # Take a snapshot and send an email
                            img_name = f"unknown_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(img_name, frame)
                            send_email_with_attachment(
                                "Unrecognized Person Detected",
                                "Several unknown people were detected.",
                                receiver_email, sender_email, sender_password,
                                img_name
                            )
                            shared_state.last_email_time = time.time()
                        # Reset the detection window
                        shared_state.unknown_face_start_time = None
                        shared_state.unknown_faces_detected = 0

            (top, right, bottom, left) = (face.top(), face.right(), face.bottom(), face.left())
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Main
if __name__ == '__main__':
    start_time = time.time()
    total_frames_processed = 0
    program_duration = 20  # Duration for which the program should run, in seconds
    
    while time.time() - start_time < program_duration:
        ret, frame = video_capture.read()
        if not ret:
            break

        total_frames_processed += 1
        thread = threading.Thread(target=process_frame, args=(frame, known_face_encodings, known_face_names))
        thread.start()
        thread.join()

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    total_duration = time.time() - start_time
    fps = total_frames_processed / total_duration if total_duration > 0 else 0

    print(f"Total known face detections: {shared_state.known_faces_count}")
    print(f"Total unknown face detections: {shared_state.unknown_faces_count}")
    print(f"Total face detections: {total_frames_processed}")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Detection speed: {fps:.2f} frames per second")
    print(f"Average faces detected per frame: {total_frames_processed / total_duration if total_duration > 0 else 0:.2f}")
    print(f"Total frames processed: {total_frames_processed}")

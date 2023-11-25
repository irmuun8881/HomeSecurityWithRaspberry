import cv2
import smtplib
from email.message import EmailMessage
import numpy as np
import dlib
import datetime
import os
import face_recognition
from dotenv import load_dotenv
from time import time

load_dotenv()

# Function to get face encodings from known images
def get_face_encodings(image_folder):
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(os.path.splitext(image_name)[0])

    return known_face_encodings, known_face_names

# Initialize dlib's face detector and face recognition model using environment variables
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(os.getenv('SHAPE_PREDICTOR_PATH'))
facerec = dlib.face_recognition_model_v1(os.getenv('FACE_RECOGNITION_MODEL_PATH'))

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known face encodings and names
images_folder = os.getenv('TRAINING_IMAGES_FOLDER')
known_face_encodings, known_face_names = get_face_encodings(images_folder)

# Email credentials and cooldown setup
sender_email = os.getenv('SENDER_EMAIL')
sender_password = os.getenv('SENDER_PASSWORD')
receiver_email = os.getenv('RECEIVER_EMAIL')
email_sent_time = None
cooldown_seconds = 15  # 15 seconds

# Initialize counters and timers
face_detections_count = 0
start_time = time()
total_frames_processed = 0
program_duration = 60  # Duration for which the program should run, in seconds

# Path for saving unknown faces
unknown_faces_dir = os.getenv('UNKNOWN_FACES_FOLDER_PATH')
os.makedirs(unknown_faces_dir, exist_ok=True)  # Create the directory if it does not exist

def send_email_notification(image_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Unrecognized Person Detected'
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg.set_content('An unrecognized person has been detected at the door.')

        with open(image_path, 'rb') as img:
            img_data = img.read()
            img_type = os.path.splitext(image_path)[1][1:]  # Extract file extension for subtype
        msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
    except Exception as e:
        print(f"Failed to send email: {e}")

# Calculate the end time based on the duration
end_time = start_time + program_duration

while time() < end_time:
    ret, frame = video_capture.read()
    if not ret:
        break  # If the frame wasn't captured correctly, stop the loop
    
    total_frames_processed += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)
    
    # Increment the face detection counter if any faces are detected
    if faces:
        face_detections_count += len(faces)

    face_encodings = []
    for face in faces:
        shape = sp(rgb_frame, face)
        face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))
        face_encodings.append(face_encoding)

        (top, right, bottom, left) = (face.top(), face.right(), face.bottom(), face.left())
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    face_names = []
    for face_encoding in face_encodings:
        distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        match = np.any(distances <= 0.45)
        name = "Unknown"
        if match:
            first_match_index = np.argmin(distances)
            name = known_face_names[first_match_index]
            print(f"Detected known person: {name}")
        else:
            print("Detected unknown person.")
            # Save the image of the unknown face
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = f"unknown_{timestamp}.jpg"
            (top, right, bottom, left) = [max(0, coord) for coord in (top, right, bottom, left)]
            face_image = frame[top:bottom, left:right]
            if face_image.size != 0:
                cv2.imwrite(os.path.join(unknown_faces_dir, filename), face_image)
            else:
                print("Error: Cropped face image is empty.")
        face_names.append(name)

    cv2.imshow('Video', frame)

    if "Unknown" in face_names:
        current_time = datetime.datetime.now()
        if email_sent_time is None or (current_time - email_sent_time).total_seconds() > cooldown_seconds:
            image_path = 'detected_person.jpg'
            cv2.imwrite(image_path, frame)
            send_email_notification(image_path)
            email_sent_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Calculations for duration and speed
total_duration = time() - start_time
detection_speed = face_detections_count / total_duration if total_duration > 0 else 0
faces_per_frame = face_detections_count / total_frames_processed if total_frames_processed > 0 else 0

# Print the results
print(f"Total face detections: {face_detections_count}")
print(f"Total processing time: {total_duration:.2f} seconds")
print(f"Detection speed: {detection_speed:.2f} faces per second")
print(f"Average faces detected per frame: {faces_per_frame:.2f}")
print(f"Total frames processed: {total_frames_processed}")

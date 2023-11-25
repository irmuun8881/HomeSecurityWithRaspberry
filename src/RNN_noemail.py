import cv2
import numpy as np
import dlib
import os
import datetime
from scipy.spatial import distance
from dotenv import load_dotenv
from time import time


# Load environment variables
load_dotenv()

# Dlib's face detection and recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(os.getenv('SHAPE_PREDICTOR_PATH'))
facerec = dlib.face_recognition_model_v1(os.getenv('FACE_RECOGNITION_MODEL_PATH'))

# Function to get face encodings from known images using dlib
def get_face_encodings(image_folder):
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            face_image = dlib.load_rgb_image(image_path)  # Using dlib to load image
            detected_faces = detector(face_image)
            for face in detected_faces:
                shape = shape_predictor(face_image, face)
                face_encoding = np.array(facerec.compute_face_descriptor(face_image, shape))
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(image_name)[0])
                break  # Assuming only one face per image

    return known_face_encodings, known_face_names

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known face encodings and names
images_folder = os.getenv('TRAINING_IMAGES_FOLDER')
known_face_encodings, known_face_names = get_face_encodings(images_folder)

# Initialize counters and timers
face_detections_count = 0
start_time = time()
total_frames_processed = 0
program_duration = 60  # Duration for which the program should run, in seconds

# Path for saving unknown faces
unknown_faces_dir = os.getenv('UNKNOWN_FACES_FOLDER_PATH')
os.makedirs(unknown_faces_dir, exist_ok=True)  # Create the directory if it does not exist

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
        shape = shape_predictor(rgb_frame, face)
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
        face_names.append(name)

    cv2.imshow('Video', frame)

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

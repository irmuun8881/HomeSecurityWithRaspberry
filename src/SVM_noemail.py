import cv2
import dlib
import numpy as np
import os
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pickle
from time import time
from dotenv import load_dotenv

load_dotenv()

# Paths to dlib models
shape_predictor_path = 'path/to/shape_predictor_68_face_landmarks.dat'
face_recognition_model_path = 'path/to/dlib_face_recognition_resnet_model_v1.dat'

# Load dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(os.getenv('SHAPE_PREDICTOR_PATH'))
facerec = dlib.face_recognition_model_v1(os.getenv('FACE_RECOGNITION_MODEL_PATH'))

# Directory containing training data
training_dir = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/training_images_folder'


# Prepare lists for encodings and labels
encodings = []
labels = []

# Loop over each person in the training directory
for person in os.listdir(training_dir):
    person_dir = os.path.join(training_dir, person)
    # Skip files that are not directories
    if not os.path.isdir(person_dir):
        continue
    # Loop over each training image for the current person
    for img_file in os.listdir(person_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            # Load the image
            img_path = os.path.join(person_dir, img_file)
            img = dlib.load_rgb_image(img_path)
            # Detect faces
            dets = detector(img, 1)
            for k, d in enumerate(dets):
                # Get the landmarks/parts for the face in box d.
                shape = sp(img, d)
                # Get the face descriptor from the 68 landmarks shape.
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                encodings.append(np.array(face_descriptor))
                labels.append(person)
                break  # Process only one face per image

# Convert labels to numerical values
le = LabelEncoder()
labels_num = le.fit_transform(labels)

# Train the SVM classifier
clf = svm.SVC(C=1.0, kernel='linear', probability=True)
clf.fit(encodings, labels_num)

print("Training complete. Starting video capture.")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize counters and timers
total_frames_processed = 0
known_faces_detected = 0
unknown_faces_detected = 0
start_time = time()
print("Entering main loop. Press 'q' to quit.")
# Real-time face recognition
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    total_frames_processed += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_frame, 1)

    for k, d in enumerate(dets):
        shape = sp(rgb_frame, d)
        face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
        face_encoding = np.array(face_descriptor)
        
        # Predict the identity using the SVM model
        prediction = clf.predict([face_encoding])
        predicted_label = le.inverse_transform(prediction)
        
        # Check prediction and increment known or unknown count
        if predicted_label[0] in labels:
            known_faces_detected += 1
            print(f"Known person detected: {predicted_label[0]}")
        else:
            unknown_faces_detected += 1
            print("Unknown person detected.")

        # Draw a rectangle with a label around the face
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, predicted_label[0], (d.left() + 6, d.bottom() - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()

# Calculations for duration and speed
total_duration = time() - start_time
faces_per_second = (known_faces_detected + unknown_faces_detected) / total_duration if total_duration > 0 else 0

# Print the results
print(f"Total frames processed: {total_frames_processed}")
print(f"Total known faces detected: {known_faces_detected}")
print(f"Total unknown faces detected: {unknown_faces_detected}")
print(f"Total processing time: {total_duration:.2f} seconds")
print(f"Detection speed: {faces_per_second:.2f} faces per second")

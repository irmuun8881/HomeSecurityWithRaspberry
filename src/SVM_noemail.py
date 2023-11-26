import cv2
import numpy as np
import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from time import time
import dlib

# Load precomputed face encodings and their labels from the pickle file
with open('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/face_encodings.pickle', 'rb') as f:
    data = pickle.load(f)

encodings = data['encodings']
labels = data['labels']

# Convert labels to numerical values using LabelEncoder
le = LabelEncoder()
labels_num = le.fit_transform(labels)

# Train the SVM classifier
clf = svm.SVC(C=1.0, kernel='linear', probability=True)
clf.fit(encodings, labels_num)

print("Training complete. Starting video capture.")

# Initialize video capture, dlib face detector, and shape predictor
video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
shape_predictor_path = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/shape_predictor_68_face_landmarks.dat'
face_recognition_model_path= '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/dlib_face_recognition_resnet_model_v1.dat'# Update this path
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_recognition_model_path)

# Initialize counters and timers
total_frames_processed = 0
known_faces_detected = 0
unknown_faces_detected = 0
start_time = time()

# Set the program duration to 20 seconds
program_duration = 20  # Run the program for 20 seconds

# Real-time face recognition
while time() < start_time + program_duration:
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

        # Increment known or unknown count
        if predicted_label[0] in labels:
            known_faces_detected += 1
            label = predicted_label[0]
            print(f"Known person detected: {label}")
        else:
            unknown_faces_detected += 1
            label = "Unknown"
            print("Unknown person detected.")

        # Draw a rectangle with a label around the face
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, label, (d.left() + 6, d.bottom() - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Print results
total_duration = time() - start_time
print(f"Total frames processed: {total_frames_processed}")
print(f"Total known faces detected: {known_faces_detected}")
print(f"Total unknown faces detected: {unknown_faces_detected}")
print(f"Total processing time: {total_duration:.2f} seconds")
print(f"Detection speed: {(known_faces_detected + unknown_faces_detected) / total_duration:.2f} faces per second")
print(f"Frames per second: {total_frames_processed / total_duration:.2f}")

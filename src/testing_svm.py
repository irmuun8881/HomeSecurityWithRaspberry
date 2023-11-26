import cv2
import numpy as np
import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import dlib
import os

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

print("Training complete.")

# Initialize dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor_path = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/shape_predictor_68_face_landmarks.dat'
facerec = dlib.face_recognition_model_v1('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/dlib_face_recognition_resnet_model_v1.dat')
sp = dlib.shape_predictor(shape_predictor_path)

# Initialize counters
total_faces_detected = 0
known_faces_detected = 0
unknown_faces_detected = 0

# Path to the testing images
test_images_dir = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/training_images_folder/negatives'  # Update with the path to your testing images

# Process each image in the test_images_dir
for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)
    if img_name.lower().endswith(('.jpg', '.png')):
        # Load the image
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        dets = detector(rgb_img, 1)
        total_faces_detected += len(dets)

        # Process each detected face
        for k, d in enumerate(dets):
            shape = sp(rgb_img, d)
            face_descriptor = facerec.compute_face_descriptor(rgb_img, shape)
            face_encoding = np.array(face_descriptor)

            # Predict the identity using the SVM model
            prediction = clf.predict([face_encoding])
            predicted_label_num = prediction[0]
            predicted_label = le.inverse_transform([predicted_label_num])[0]

            # Update counters based on whether the face is known, unknown, or a negative
            if predicted_label == 'negative':
                unknown_faces_detected += 1
                print(f"Image {img_name}: Negative sample detected.")
            elif predicted_label == 'negatives':
                unknown_faces_detected += 1
                print(f"Image {img_name}: Unknown person detected.")
            elif predicted_label in labels:
                known_faces_detected += 1
                print(f"Image {img_name}: Known person detected - {predicted_label}")
            else:
                unknown_faces_detected += 1
                print(f"Image {img_name}: Unknown person detected.")

# Print final results
print(f"Total faces processed: {total_faces_detected}")
print(f"Total known faces detected: {known_faces_detected}")
print(f"Total unknown faces detected: {unknown_faces_detected}")

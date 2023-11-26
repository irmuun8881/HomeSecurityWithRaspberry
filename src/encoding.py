import os
import pickle
import dlib
import numpy as np

# Set your paths here
shape_predictor_path = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/shape_predictor_68_face_landmarks.dat'
face_recognition_model_path = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/dlib_face_recognition_resnet_model_v1.dat'
training_images_dir = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/training_images_folder'

# Initialize dlib's face detector and the face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_recognition_model_path)

# Prepare lists for encodings and labels
encodings = []
labels = []

# Loop over each person in the training images directory
for person_name in os.listdir(training_images_dir):
    person_dir = os.path.join(training_images_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    # Loop over each training image for the current person
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(('.jpg', '.png')):
            print(f"Processing file: {img_file}")
            img_path = os.path.join(person_dir, img_file)
            img = dlib.load_rgb_image(img_path)
            
            # Detect faces in the image
            detected_faces = detector(img, 1)
            if len(detected_faces) == 0:
                print(f"No faces found in {img_file} - Skipping.")
                continue

            # We assume that each image has one face only
            shape = sp(img, detected_faces[0])
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            encodings.append(np.array(face_descriptor))
            labels.append(person_name)

# Save encodings and labels to a file in the current directory
with open('face_encodings.pickle', 'wb') as f:
    pickle.dump({'encodings': encodings, 'labels': labels}, f)
print("Encodings saved to face_encodings.pickle")

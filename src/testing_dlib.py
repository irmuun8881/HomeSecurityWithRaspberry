import cv2
import numpy as np
import dlib
import os
from scipy.spatial import distance

# Load dlib's face detection and recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/dlib_face_recognition_resnet_model_v1.dat')

# Function to get face encodings from known images using dlib
def get_face_encodings(image_folder):
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            face_image = dlib.load_rgb_image(image_path)
            detected_faces = detector(face_image)
            for face in detected_faces:
                shape = shape_predictor(face_image, face)
                face_encoding = np.array(facerec.compute_face_descriptor(face_image, shape))
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(image_name)[0])
                break
    return known_face_encodings, known_face_names

# Load known face encodings and names
images_folder = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/training_images_folder/irmuun' 
known_face_encodings, known_face_names = get_face_encodings(images_folder)

# Function to process and test each image in the folder
def test_images_in_folder(folder_path):
    total_faces_detected = 0
    total_known_faces = 0
    total_unknown_faces = 0

    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(folder_path, image_name)
            image = dlib.load_rgb_image(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            faces = detector(rgb_image)
            total_faces_detected += len(faces)
            print(f"Processing {image_name}, detected {len(faces)} faces.")

            for face in faces:
                shape = shape_predictor(rgb_image, face)
                face_encoding = np.array(facerec.compute_face_descriptor(rgb_image, shape))

                distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                match = np.any(distances <= 0.45)
                if match:
                    total_known_faces += 1
                    first_match_index = np.argmin(distances)
                    name = known_face_names[first_match_index]
                    print(f"Detected known person: {name}")
                else:
                    total_unknown_faces += 1
                    print("Detected unknown person.")

    print(f"Total faces detected: {total_faces_detected}")
    print(f"Total known faces: {total_known_faces}")
    print(f"Total unknown faces: {total_unknown_faces}")

# Specify the folder to test
test_images_folder = '/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/testing/unknown_faces_folder'  # Update this path
test_images_in_folder(test_images_folder)

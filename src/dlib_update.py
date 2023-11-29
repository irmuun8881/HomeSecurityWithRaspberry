import cv2
import numpy as np
import dlib
import pickle
from threading import Thread
import time

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
video_capture.set(cv2.CAP_PROP_FPS, 10)  # Reduced frame rate

# Function to process each frame
def process_frame(frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)
    for face in faces:
        shape = shape_predictor(rgb_frame, face)
        face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))

        # Compare face encodings
        distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        best_match_index = np.argmin(distances)
        name = "Unknown"
        if distances[best_match_index] < 0.5:  # Threshold for recognizing a face
            name = known_face_names[best_match_index]

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
        thread = Thread(target=process_frame, args=(frame, known_face_encodings, known_face_names))
        thread.start()
        thread.join()

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    total_duration = time.time() - start_time
    fps = total_frames_processed / total_duration if total_duration > 0 else 0

    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Frames Per Second: {fps:.2f}")
    print(f"Total frames processed: {total_frames_processed}")

import cv2
import dlib
import numpy as np
import pickle
import queue
import threading
import time

# Load pre-existing encodings
with open("/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/known_faces.pkl", "rb") as f:
    known_face_encodings, known_face_labels = pickle.load(f)

# Initialize Haar Cascade, dlib's Face Recognition, and CSRT Tracker
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = dlib.face_recognition_model_v1('/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/dlib_face_recognition_resnet_model_v1.dat')
predictor_path = "/Users/JessFort/Documents/My_Coding_folder/IOT_Oke/models/shape_predictor_68_face_landmarks.dat"  # Update with the correct path
shape_predictor = dlib.shape_predictor(predictor_path)
tracker = cv2.TrackerCSRT_create()

# Function to compute face encoding using dlib
def get_face_encoding(face_image):
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    dlib_image = dlib.get_frontal_face_detector()(face_image_rgb, 1)
    if dlib_image:
        shape = shape_predictor(face_image_rgb, dlib_image[0])
        face_descriptor = face_recognizer.compute_face_descriptor(face_image_rgb, shape)
        return np.array(face_descriptor)
    return None

# Function to compare face encodings
def compare_faces(known_encodings, face_encoding_to_check):
    if face_encoding_to_check is not None:
        distances = np.linalg.norm(known_encodings - face_encoding_to_check, axis=1)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.5:  # Threshold for recognizing a face
            return known_face_labels[best_match_index]
    return "Unknown"

# Thread for processing frames
def process_frame(frame_queue, tracked_queue, face_count):
    while True:
        if not frame_queue.empty():
            frame, gray, (x, y, w, h) = frame_queue.get()
            if frame is not None:
                face_img = frame[y:y+h, x:x+w]
                encoding = get_face_encoding(face_img)
                label = compare_faces(known_face_encodings, encoding)
                print(f"Detected: {label}")  # Print detected person status
                tracker.init(frame, (x, y, w, h))
                tracked_queue.put((label, (x, y, w, h)))
                face_count[0] += 1

# Main
if __name__ == '__main__':
    frame_queue = queue.Queue()
    tracked_queue = queue.Queue()
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Camera is not available")
        exit()

    face_count = [0]  # Using a list for mutable integer
    threading.Thread(target=process_frame, args=(frame_queue, tracked_queue, face_count), daemon=True).start()

    start_time = time.time()
    frame_count = 0
    tracking = False

    while time.time() - start_time < 20:  # Run for 20 seconds
        ret, frame = video_capture.read()
        if not ret or frame is None:
            continue  # Skip processing if the frame is empty

        frame = cv2.resize(frame, (320, 240))
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # If tracking is active
        if tracking:
            # Check if the frame is not empty before updating the tracker
            if frame is not None and frame.any():
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    tracking = False
        else:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                frame_queue.put((frame, gray, faces[0]))
                # Initialize the tracker with the first detected face
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tracking = True

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Calculate and print statistics
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    faces_per_second = face_count[0] / total_time

    print(f"Total Frames: {frame_count}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Frames Per Second: {fps:.2f}")
    print(f"Total Faces Detected: {face_count[0]}")
    print(f"Faces Per Second: {faces_per_second:.2f}")

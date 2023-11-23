import cv2
import smtplib
from email.message import EmailMessage
import numpy as np
import dlib
import datetime
import os
import face_recognition
from dotenv import load_dotenv

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
images_folder = "/Users/JessFort/Documents/My_Coding_folder/IOT_project/image_folder"  # Update with the path to your images folder
known_face_encodings, known_face_names = get_face_encodings(images_folder)

# Email sending cooldown setup
email_sent_time = None
cooldown_minutes = 5

# Email credentials from environment variables
sender_email = os.getenv('SENDER_EMAIL')
sender_password = os.getenv('SENDER_PASSWORD')
receiver_email = os.getenv('RECEIVER_EMAIL')

def send_email_notification(image_path):
    # Make sure the contents of this function are indented
    msg = EmailMessage()
    msg['Subject'] = 'Unrecognized Person Detected'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content('An unrecognized person has been detected at the door.')

    with open(image_path, 'rb') as img:
        file_data = img.read()
        file_type = 'jpg'
        file_name = image_path

    msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(rgb_frame)
    face_encodings = []

    for face in faces:
        shape = sp(rgb_frame, face)
        face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))
        face_encodings.append(face_encoding)

    face_names = []
    for face_encoding in face_encodings:
        # Compare face encodings with known faces
        if known_face_encodings:
            distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
            match = np.any(distances <= 0.6)
            name = "Unknown"
            if match:
                first_match_index = np.argmin(distances)
                name = known_face_names[first_match_index]
        else:
            name = "Unknown"
        face_names.append(name)

    if "Unknown" in face_names:
        current_time = datetime.datetime.now()
        if email_sent_time is None or (current_time - email_sent_time).total_seconds() > cooldown_minutes * 60:
            cv2.imwrite('unrecognized_person.jpg', frame)
            send_email_notification('unrecognized_person.jpg')
            email_sent_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
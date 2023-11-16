import cv2
import smtplib
import imghdr
from email.message import EmailMessage
import face_recognition

cap = cv2.VideoCapture(0)
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

threshold_area = 5000
notification_sent = False

sender_email = "your_email@gmail.com"
sender_password = "your_password"
receiver_email = "recipient_email@example.com"

def send_email_notification():
    msg = EmailMessage()
    msg['Subject'] = 'Person Detected at the Door'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content('A person has been detected at the door!')

    with open('person_detected.jpg', 'rb') as f:
        file_data = f.read()
        file_type = imghdr.what(f.name)
        file_name = f.name

    msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

known_family_members = ["Alice", "Bob", "Charlie"]  # Example family members

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in pedestrians:
        if x < 400:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if not notification_sent:
                cv2.imwrite('person_detected.jpg', frame)
                detected_person = "Unknown"

                # Face recognition logic
                unknown_face_encoding = face_recognition.face_encodings(frame)[0]
                results = face_recognition.compare_faces(known_encodings, unknown_face_encoding)
                if not any(results):  # If no match found
                    send_email_notification()
                    notification_sent = True

    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

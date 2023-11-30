Smart Home Security System using Raspberry Pi
Project Overview
This project develops a Smart Home Security System using Raspberry Pi 3b+ and OpenCV. It focuses on real-time facial recognition to differentiate between known individuals and potential intruders, sending immediate email alerts upon detecting unknown faces.

Features
Facial Recognition: Leverages dlib's model for accurate identification.
Real-Time Notifications: Sends email alerts for unrecognized faces.
Optimized for Raspberry Pi: Efficient for Raspberry Pi's limited processing capacity.

Hardware Requirements
Raspberry Pi (3b+ recommended)
Camera Module
Internet Connection (WIFI/Ethernet)
MicroSD Card with Raspberry Pi OS
Power Supply
HDMI Cable and Monitor (initial setup)
Raspberry Pi Case
Software Requirements
Raspberry Pi OS
Python 3.6+
OpenCV
dlib
Email notification system (e.g., smtplib, Pushbullet, Twilio)
Environment Setup and Installation
1. Raspberry Pi Setup:
Install Raspberry Pi OS on the MicroSD card.
Attach the camera module and connect the Raspberry Pi to the internet.
2. Install Dependencies:
bash
Copy code
sudo apt update
sudo apt upgrade
pip install -r requirements.txt
3. Configuration for Email Notifications:
Set up an SMTP server or use services like Pushbullet/Twilio.
4. Clone Repository and Modify Paths:
bash
Copy code
git clone https://github.com/irmuun8881/HomeSecurityWithRaspberry/.git
cd [Repository Directory]
Update file paths in the scripts to point to your local directories (e.g., for saved models, known faces data).
5. Running the Application:
bash
Copy code
python [Script Name]
Usage
Position the device strategically (e.g., near the front door).
The system processes the video feed, recognizes faces, and categorizes them as known or unknown.
Unrecognized faces trigger an email notification.
Additional Notes
Ensure your environment is correctly set up with all dependencies.
Regularly update the system for security and functionality improvements.
Project Results
The system effectively recognizes faces, providing a secure environment. Detailed results and analytics can be found in the results folder.

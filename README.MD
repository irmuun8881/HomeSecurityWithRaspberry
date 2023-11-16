# Raspberry Pi Door Monitoring System with Face Recognition

This project utilizes a Raspberry Pi, a webcam, and OpenCV for monitoring activity at a door. It detects individuals approaching the door using a pre-trained Haar cascade for pedestrian detection and performs face recognition to determine if the detected person is a known family member.

## Prerequisites

- Python 3.x
- OpenCV
- imghdr
- secure-smtplib
- face_recognition

## Installation

1. Clone this repository.
2. Install the required libraries using the provided requirements file:
   ```bash
   pip install -r requirements.txt

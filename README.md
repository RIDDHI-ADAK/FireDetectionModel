# FireDetection App

Welcome to the **FireDetection** app! This application leverages Streamlit to provide real-time fire detection capabilities using machine learning and computer vision. The goal is to help prevent fire hazards and assist in timely emergency responses.



## 📋 Features

- **Real-time Fire Detection**: Detect fire using detected images or live video.
- **Fast and Accurate**: Uses machine learning models to identify potential fire hazards.
- **User-friendly Interface**: Intuitive UI to easily detected images and get results.
- **Interactive Results**: View detection outputs with confidence scores and fire zones.

---










# Table of Contents
Introduction

Features

Technologies Used

Setup Instructions

Usage

Error Handling

Future Enhancements

Contact

# Introduction
The Fire Detection Project is designed to detect fires in real-time using a combination of hardware and machine learning. It uses a camera for image-based fire detection, smoke and temperature sensors for environmental monitoring, and an Arduino board to interface with the sensors. The system sends alerts via SMS when a fire is detected.

# Features
Real-time fire detection using image classification with Convolutional Neural Networks (CNN).

Smoke and temperature monitoring using MQ2 and DHT11 sensors.

Automated SMS alerts during fire detection using Twilio.

Data storage and visualization in the web application.

Separate display pages for fire data and sensor data.

Technologies Used
# Hardware:

Arduino Uno Board (with DHT11 Temperature Sensor)

MQ2 Smoke Sensor

DHT11 Temperature Sensor

Camera for Fire Image Detection

# Software:

Backend: SQLite (or any preferred database)

Frontend: Streamlit

Machine Learning: TensorFlow (CNN for image classification)

SMS Service: Twilio

# Setup Instructions
1. Prerequisites:
Install Python 3.10.0 .

Ensure Arduino is connected to your computer.

2. Clone the Repository:
bash
Copy
git clone [repository-url]
cd [repository-name]
3. Install Required Packages:
bash
Copy
 'pip install streamlit mysql-connector-python pyserial tensorflow numpy pandas opencv-python pyttsx3 twilio gdown'
4. Arduino Setup:
Open the provided Arduino code file in the Arduino IDE.

Upload the code to the Arduino Uno board.

5. Run the Web App:
bash
Copy
streamlit run W.py
Usage
Open the web application in your browser (usually at http://localhost:8501).

Use the sidebar to view:

Fire Data: Displays time, smoke level, temperature, and fire detection status (0 = No Fire, 1 = Fire).

Sensor Display Data: Shows the latest smoke and temperature data when no fire is detected.

Error Handling
Ensure all required fields are filled before submitting data.

Verify that the Arduino is properly connected before running the app.

Check for correct Twilio setup to ensure SMS alerts are sent.

Future Enhancements
User Authentication: Support for multiple users.

Task Prioritization: Features to prioritize tasks.

Notifications: Alerts for upcoming due dates.

UI/UX Improvements: Enhanced user interface and experience.

# Twilio SMS Setup
To enable SMS alerts:

Create a Twilio Account: Twilio Signup

Get your account_sid, auth_token, and twilio_number.

Update the W.py file:

python
Copy
# Twilio Account Setup
account_sid = 'YOUR_ACCOUNT_SID'
auth_token = 'YOUR_AUTH_TOKEN'
twilio_number = 'YOUR_TWILIO_NUMBER'
Set recipient numbers:

python
Copy
recipient_number1 = '+917895467834'
recipient_number2 = '+917253715211'
recipient_number3 = '+915646546555'
recipient_number4 = '+914565852425'


# Software Development
The software for the Fire Detection Project is developed to integrate hardware components (camera, smoke sensor, temperature sensor, and Arduino board) with machine learning algorithms for real-time fire detection. The software includes:

Backend Development: Handling data storage, sensor communication, and real-time processing using SQLite.

Frontend Development: A user-friendly web interface built with Streamlit for visualizing fire data and sensor statistics.

Machine Learning Integration: Implementation of Convolutional Neural Networks (CNN) for accurate fire detection through image classification.

Alert System: Automated SMS alerts using Twilio when a fire is detected.

The software is designed to be scalable and can be enhanced with additional features in the future. Future improvements will focus on optimizing performance, adding advanced analytics, and expanding the system’s capabilities for better fire detection and user experience.


# Contact
For any questions, feedback, or suggestions, feel free to contact me via:
GitHub: https://github.com/DONCHAN70047

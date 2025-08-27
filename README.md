# Drowsiness Detection System
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

## Description

The **Drowsiness Detection System** is a real-time web application that uses computer vision and machine learning to monitor a user's facial features and detect signs of drowsiness while driving. The system tracks eye aspect ratio (EAR) and lip distance (for yawning detection). When the system detects potential drowsiness or fatigue, it triggers an alert to warn the driver. The app also includes user authentication and logs detection sessions for tracking.

## Features

- **User Authentication**: Login and signup functionality to track sessions.
- **Drowsiness Detection**: Real-time detection of drowsiness and yawning using webcam feed.
- **Alert System**: Audio alerts for drowsiness and yawning detection.
- **Session Logging**: Logs each user’s login, session start, and session end in the database for tracking.
- **Responsive Dashboard**: Provides real-time webcam feed and detection status.

## Technologies Used

- **Python**: Programming language used for backend logic.
- **Flask**: A micro web framework to build the web app.
- **OpenCV**: For capturing webcam feed and processing images.
- **dlib**: Used for facial landmark detection and eye aspect ratio calculation.
- **PostgreSQL**: Database used for storing user data and session logs.
- **HTML, CSS**: Frontend for rendering the user interface.
- **SQLite**: For local testing during development (switchable to PostgreSQL for production).

---

## Installation

### Prerequisites

Before you can run this application, make sure you have the following installed:
- **Python 3**: [Download and Install Python](https://www.python.org/downloads/)
- **PostgreSQL** (for production) or SQLite (for local testing)
- **Pip** (Python package manager) for installing dependencies.

### Steps to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection

# Hand Gesture Spotify Control

This project allows you to control Spotify playback using hand gestures captured via a webcam. It leverages computer vision (OpenCV and MediaPipe) to detect hand landmarks and the Spotify Web API (via Spotipy) to manage playback, volume, and track skipping. Gestures are intuitive and include play/pause, volume adjustment, track skipping, and enabling/disabling controls.

## Features
- **Play/Pause**: Toggle playback with a simple gesture.
- **Volume Control**: Adjust volume based on thumb-to-index finger distance.
- **Track Skipping**: Skip forward or backward by moving your index finger.
- **Toggle Controls**: Enable or disable gesture controls with a specific hand pose.
- **Real-Time Feedback**: Visual feedback on the camera feed shows playback state, volume, and FPS.

## Prerequisites
- Python 3.7 or higher
- A webcam (default camera index 0)
- A Spotify account and an active Spotify client running on a device (e.g., desktop or phone)
- Spotify Developer credentials (Client ID and Client Secret)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-spotify-control.git
   cd hand-gesture-spotify-control

2. **Install Dependencies**
   pip install spotipy opencv-python numpy mediapipe

3. **Set up Spotify API in Spotify Dashboard**
   -Create an app to get Client ID and Secret ID
   -Replace clientid and Secret within the python files

5. 

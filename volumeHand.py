# Import required libraries
import spotipy  # Spotify API wrapper
from spotipy.oauth2 import SpotifyOAuth  # Authentication handler for Spotify
import cv2 as cv  # OpenCV for computer vision
import numpy as np  # Numerical operations
import time  # Time-related functions
import handTrackingModule as htm  # Custom hand tracking module
from spotipy.exceptions import SpotifyException  # Spotify-specific exceptions
import threading  # Threading support
from concurrent.futures import ThreadPoolExecutor  # Thread pool for parallel execution

# Constants for camera and control timing
CAM_WIDTH, CAM_HEIGHT = 640, 480  # Camera resolution
VOLUME_UPDATE_INTERVAL = 0.1  # Minimum time between volume updates (seconds)
SKIP_FRAME_COUNT = 10  # Number of frames to track for skip gesture
SKIP_THRESHOLD = 40  # Pixel threshold for skip detection
SKIP_COOLDOWN = 1.0  # Cooldown between skip actions (seconds)
SKIP_CONFIRM_DURATION = 0.5  # Time to confirm skip gesture (seconds)
TOGGLE_DURATION = 1.0  # Time to hold toggle gesture (seconds)
PLAY_PAUSE_COOLDOWN = 2.5  # Cooldown between play/pause actions (seconds)

def initialize_spotify():
    """Initialize Spotify client with OAuth authentication."""
    scope = "user-modify-playback-state user-read-playback-state"  # Required Spotify permissions
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id="client ID",  # Spotify app client ID
        client_secret="client secret",  # Spotify app secret
        redirect_uri="http://localhost:8888/callback",  # Callback URI for auth
        scope=scope
    ))

def initialize_camera():
    """Set up webcam with specified resolution."""
    cap = cv.VideoCapture(0)  # Open default camera
    cap.set(3, CAM_WIDTH)  # Set width
    cap.set(4, CAM_HEIGHT)  # Set height
    return cap

def detect_hand_landmarks(img, detector):
    """Detect and draw hand landmarks on the image."""
    img = detector.findHands(img, draw=True)  # Process image for hand detection
    return img, detector.findPosition(img, draw=False)  # Return image and landmark positions

def get_extended_fingers(lm_list):
    """Determine which fingers are extended based on landmark positions."""
    extended = [False] * 5  # Array for 5 fingers
    wrist = np.array(lm_list[0][1:3])  # Wrist position
    # Check each finger (thumb, index, middle, ring, pinky)
    for i, tip_id, mcp_id in zip(range(5), [4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
        tip = np.array(lm_list[tip_id][1:3])  # Finger tip position
        mcp = np.array(lm_list[mcp_id][1:3])  # Middle joint position
        # Finger is extended if tip is significantly further from wrist than MCP
        if np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.2:
            extended[i] = True
    return extended

# Spotify API call wrappers for threading
def spotify_play_pause(sp, is_playing):
    """Thread-safe wrapper for toggling playback state."""
    try:
        if is_playing:
            sp.pause_playback()
            print("Playback paused (threaded)")
        else:
            sp.start_playback()
            print("Playback started (threaded)")
    except SpotifyException as e:
        print(f"Spotify error in play/pause: {e}")

def spotify_adjust_volume(sp, vol_level):
    """Thread-safe wrapper for volume adjustment."""
    try:
        sp.volume(vol_level)
        print(f"Volume set to {vol_level}% (threaded)")
    except SpotifyException as e:
        if e.http_status == 429:  # Handle rate limiting
            print("Rate limit hit in volume adjust, backing off...")
            time.sleep(1)
        else:
            print(f"Spotify error in volume: {e}")

def spotify_skip_track(sp, direction):
    """Thread-safe wrapper for skipping tracks."""
    try:
        if direction == "next":
            sp.next_track()
            print("Next track (threaded)")
        elif direction == "previous":
            sp.previous_track()
            print("Previous track (threaded)")
    except SpotifyException as e:
        print(f"Spotify error in skip: {e}")

def play_pause(sp, lm_list, last_play_pause_time, img, executor, playback_state):
    """Handle play/pause gesture detection and execution."""
    current_time = time.time()
    if current_time - last_play_pause_time < PLAY_PAUSE_COOLDOWN:  # Check cooldown
        return last_play_pause_time, playback_state

    extended = get_extended_fingers(lm_list)
    # Play/pause gesture: index and middle fingers extended
    is_play_pause_gesture = (extended[1] and extended[2] and 
                            not extended[0] and not any(extended[3:5]))

    if not is_play_pause_gesture:
        return last_play_pause_time, playback_state

    try:
        playback = sp.current_playback()  # Get current playback state
        is_playing = playback is not None and playback.get('is_playing', False)
        executor.submit(spotify_play_pause, sp, is_playing)  # Execute in thread
        return current_time, not is_playing  # Update state optimistically
    except SpotifyException as e:
        print(f"Spotify error: {e}")
        return last_play_pause_time, playback_state

def adjust_volume(sp, lm_list, last_volume_time, last_volume, img, executor):
    """Adjust Spotify volume based on thumb-index finger distance."""
    current_time = time.time()
    if current_time - last_volume_time < VOLUME_UPDATE_INTERVAL:  # Rate limit
        return last_volume_time, last_volume

    thumb_tip = np.array(lm_list[4][1:3])  # Thumb tip position
    index_tip = np.array(lm_list[8][1:3])  # Index tip position
    distance = np.linalg.norm(thumb_tip - index_tip)  # Distance between fingers
    vol_level = int(np.interp(distance, [20, 200], [0, 100]))  # Map to 0-100%

    executor.submit(spotify_adjust_volume, sp, vol_level)  # Execute in thread
    draw_volume_feedback(img, thumb_tip, index_tip, vol_level)  # Visual feedback
    return current_time, vol_level  # Update state optimistically

def handle_song_skip(sp, lm_list, skip_data, last_skip_time, img, executor):
    """Detect and handle track skipping gestures."""
    current_time = time.time()
    if current_time - last_skip_time < SKIP_COOLDOWN:  # Check cooldown
        return skip_data, last_skip_time

    index_tip = np.array(lm_list[8][1:3])  # Index finger tip position
    history, start_time = skip_data  # Unpack skip tracking data

    if start_time is None:
        start_time = current_time  # Start tracking gesture

    history.append(index_tip[0])  # Track x-position
    if len(history) > SKIP_FRAME_COUNT:
        history.pop(0)  # Maintain fixed history size

    # Draw tracking line
    if len(history) > 1:
        for i in range(1, len(history)):
            cv.line(img, (int(history[i-1]), index_tip[1]), 
                   (int(history[i]), index_tip[1]), (0, 255, 255), 2)
        cv.putText(img, "Tracking skip...", (50, 130), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if current_time - start_time < SKIP_CONFIRM_DURATION:  # Wait for confirmation
        return (history, start_time), last_skip_time

    if len(history) == SKIP_FRAME_COUNT:  # Full history collected
        avg_start = sum(history[:SKIP_FRAME_COUNT//2]) / (SKIP_FRAME_COUNT//2)
        avg_end = sum(history[SKIP_FRAME_COUNT//2:]) / (SKIP_FRAME_COUNT//2)
        displacement = avg_end - avg_start  # Calculate movement

        if displacement > SKIP_THRESHOLD:  # Skip forward
            executor.submit(spotify_skip_track, sp, "next")
            return ([], None), current_time
        elif displacement < -SKIP_THRESHOLD:  # Skip backward
            executor.submit(spotify_skip_track, sp, "previous")
            return ([], None), current_time

    return (history, start_time), last_skip_time

def draw_volume_feedback(img, thumb_tip, index_tip, vol_level):
    """Draw visual feedback for volume adjustment."""
    thumb_pos = tuple(thumb_tip.astype(int))
    index_pos = tuple(index_tip.astype(int))
    cv.circle(img, thumb_pos, 15, (255, 0, 0), cv.FILLED)  # Thumb marker
    cv.circle(img, index_pos, 15, (255, 0, 0), cv.FILLED)  # Index marker
    cv.line(img, thumb_pos, index_pos, (255, 255, 255), 2)  # Connecting line
    cv.putText(img, f"Volume: {int(vol_level)}%", (50, 70),  # Volume text
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def toggle_controls(extended, controls_active, toggle_start_time):
    """Toggle control state with thumb and pinky gesture."""
    toggle_gesture = extended[0] and extended[4] and not any(extended[1:4])
    current_time = time.time()

    if toggle_gesture:
        if toggle_start_time is None:
            toggle_start_time = current_time  # Start timing gesture
        elif current_time - toggle_start_time > TOGGLE_DURATION:  # Gesture held long enough
            controls_active = not controls_active
            print("Controls toggled to", "active" if controls_active else "inactive")
            toggle_start_time = None
    else:
        toggle_start_time = None  # Reset if gesture not detected

    return controls_active, toggle_start_time

def display_status(img, controls_active, toggle_start_time, fps, playback_state):
    """Display current system status on the image."""
    status_text = "Controls: Active" if controls_active else "Controls: Inactive"
    status_color = (0, 255, 0) if controls_active else (0, 0, 255)
    cv.putText(img, status_text, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    playback_text = "Playing" if playback_state else "Paused"
    playback_color = (0, 255, 0) if playback_state else (255, 0, 0)
    cv.putText(img, playback_text, (50, 160), cv.FONT_HERSHEY_SIMPLEX, 1, playback_color, 2)

    if toggle_start_time is not None:
        cv.putText(img, "Toggling controls...", (50, 130),
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv.putText(img, f'FPS: {int(fps)}', (40, 50),  # Display frame rate
              cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)

def main():
    """Main execution loop for hand gesture Spotify control."""
    sp = initialize_spotify()  # Initialize Spotify client
    cap = initialize_camera()  # Set up camera
    detector = htm.handDetector(detectionCon=0.7)  # Initialize hand detector

    # Use thread pool for non-blocking Spotify API calls
    with ThreadPoolExecutor(max_workers=3) as executor:
        controls_active = True  # Start with controls enabled
        toggle_start_time = None  # Toggle gesture timer
        last_volume_time = 0  # Last volume adjustment time
        last_volume = 0  # Last volume level
        last_skip_time = 0  # Last skip time
        last_play_pause_time = 0  # Last play/pause time
        playback_state = False  # Current playback state
        skip_data = ([], None)  # Skip gesture tracking data
        prev_time = 0  # For FPS calculation

        while True:
            success, img = cap.read()  # Capture frame
            if not success:
                break

            img, lm_list = detect_hand_landmarks(img, detector)  # Process hand landmarks
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
            prev_time = current_time

            if lm_list:  # If hand detected
                extended = get_extended_fingers(lm_list)  # Get finger states
                controls_active, toggle_start_time = toggle_controls(extended, controls_active, toggle_start_time)

                if controls_active:  # Process gestures if controls active
                    if all(extended):  # All fingers extended: volume control
                        last_volume_time, last_volume = adjust_volume(sp, lm_list, last_volume_time, last_volume, img, executor)
                        skip_data = ([], None)  # Reset skip tracking
                    elif extended[1] and not extended[0] and not any(extended[2:]):  # Index only: skip
                        skip_data, last_skip_time = handle_song_skip(sp, lm_list, skip_data, last_skip_time, img, executor)
                    elif extended[1] and extended[2] and not extended[0] and not any(extended[3:5]):  # Index+middle: play/pause
                        last_play_pause_time, playback_state = play_pause(sp, lm_list, last_play_pause_time, img, executor, playback_state)
                        skip_data = ([], None)  # Reset skip tracking
                    else:
                        skip_data = ([], None)  # Reset skip tracking if no gesture matched

            display_status(img, controls_active, toggle_start_time, fps, playback_state)  # Update display
            cv.imshow("img", img)  # Show processed frame
            if cv.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' press
                break

    cap.release()  # Release camera
    cv.destroyAllWindows()  # Close windows

if __name__ == "__main__":
    main()  # Run the program

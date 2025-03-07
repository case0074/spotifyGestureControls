# Import required libraries
import spotipy  # Spotify Web API wrapper for controlling playback
from spotipy.oauth2 import SpotifyOAuth  # OAuth authentication for Spotify API
import cv2 as cv  # OpenCV for camera input and image processing
import numpy as np  # NumPy for numerical operations (e.g., distance calculations)
import time  # Time module for timing and cooldowns
import handTrackingModule as htm  # Custom module for hand landmark detection (likely MediaPipe-based)
from spotipy.exceptions import SpotifyException  # Exception handling for Spotify API errors

# Define constants for configuration
CAM_WIDTH, CAM_HEIGHT = 640, 480  # Camera resolution in pixels
VOLUME_UPDATE_INTERVAL = 0.1  # Minimum time (seconds) between volume updates to avoid API spam
SKIP_FRAME_COUNT = 10  # Number of frames to track for song skip gesture
SKIP_THRESHOLD = 40  # Pixel displacement threshold for detecting a skip
SKIP_COOLDOWN = 1.0  # Minimum time (seconds) between skip actions
SKIP_CONFIRM_DURATION = 0.5  # Time (seconds) to hold skip gesture before triggering
TOGGLE_DURATION = 1.0  # Time (seconds) to hold toggle gesture to enable/disable controls
PLAY_PAUSE_COOLDOWN = 2.5  # Minimum time (seconds) between play/pause toggles to prevent rapid switching

#landmark ID's
# 0 = Base of wrist
# 1 = Thumb CMC (start of thumb / base of thumb)
# 2 = Thumb MCP (middle of thumb / joint near palm)
# 3 = Thumb IP (joint before thumb tip)
# 4 = Thumb tip
# 5 = Index MCP (start of index finger)
# 6 = Index PIP (middle joint of index finger)
# 7 = Index DIP (joint before index tip)
# 8 = Index tip
# 9 = Middle MCP (start of middle finger)
# 10 = Middle PIP (middle joint of middle finger)
# 11 = Middle DIP (joint before middle tip)
# 12 = Middle tip
# 13 = Ring MCP (start of ring finger)
# 14 = Ring PIP (middle joint of ring finger)
# 15 = Ring DIP (joint before ring tip)
# 16 = Ring tip
# 17 = Pinky MCP (start of pinky)
# 18 = Pinky PIP (middle joint of pinky)
# 19 = Pinky DIP (joint before pinky tip)
# 20 = Pinky tip





# Spotify setup function
def initialize_spotify():
    """Initialize Spotify API with OAuth authentication."""
    # Define scopes for reading playback state and modifying playback
    scope = "user-modify-playback-state user-read-playback-state"
    # Return authenticated Spotify client with provided credentials
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id="client id",  # Unique ID for your Spotify app
        client_secret="secret id",  # Secret key for your Spotify app
        redirect_uri="http://localhost:8888/callback",  # URI for OAuth redirect
        scope=scope  # Permissions requested from the user
    ))

# Camera initialization function
def initialize_camera():
    """Set up the camera with specified dimensions."""
    cap = cv.VideoCapture(0)  # Open default camera (index 0)
    cap.set(3, CAM_WIDTH)  # Set width of camera feed
    cap.set(4, CAM_HEIGHT)  # Set height of camera feed
    return cap  # Return camera object for frame capture

# Hand landmark detection function
def detect_hand_landmarks(img, detector):
    """Detect hand landmarks in the image."""
    img = detector.findHands(img, draw=True)  # Process image and draw hand landmarks
    # Return processed image and list of landmark positions (id, x, y, z)
    return img, detector.findPosition(img, draw=False)

# Gesture recognition function
def get_extended_fingers(lm_list):
    """Determine which fingers are extended based on landmark positions."""
    # Initialize list assuming all fingers are folded (not extended)
    extended = [False] * 5  # [thumb, index, middle, ring, pinky]
    # Convert wrist coordinates (landmark 0) to a 2D NumPy array for vector operations
    wrist = np.array(lm_list[0][1:3])  # [x, y] of wrist, ignoring z
    # Loop over each finger, pairing its index with tip and MCP joint IDs
    for i, tip_id, mcp_id in zip(range(5), [4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
        # Tip IDs: thumb=4, index=8, middle=12, ring=16, pinky=20
        # MCP IDs: thumb=2, index=5, middle=9, ring=13, pinky=17
        tip = np.array(lm_list[tip_id][1:3])  # 2D coordinates of fingertip [x, y]
        mcp = np.array(lm_list[mcp_id][1:3])  # 2D coordinates of MCP joint [x, y]
        # Compare distances: if tip-to-wrist > 1.2 * mcp-to-wrist, finger is extended
        if np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.2:
            extended[i] = True  # Mark finger as extended
    return extended  # Return list of booleans indicating extended fingers

# Play/pause toggle function
def play_pause(sp, lm_list, last_play_pause_time, img):
    """Toggle Spotify playback and return updated time and state."""
    current_time = time.time()  # Get current timestamp
    # Check if enough time has passed since last toggle to avoid rapid switching
    if current_time - last_play_pause_time < PLAY_PAUSE_COOLDOWN:
        return last_play_pause_time, None  # No change if within cooldown

    extended = get_extended_fingers(lm_list)  # Detect extended fingers
    # Play/pause gesture: index and middle extended, others folded
    is_play_pause_gesture = (extended[1] and extended[2] and 
                             not extended[0] and not any(extended[3:5]))

    if not is_play_pause_gesture:
        return last_play_pause_time, None  # No action if gesture not detected

    try:
        playback = sp.current_playback()  # Get current playback state from Spotify
        # If no playback or paused, start playing
        if playback is None or not playback.get('is_playing', False):
            sp.start_playback()  # Start playback on active device
            print("Playback started")
            return current_time, True  # Return new time and playing state
        else:
            sp.pause_playback()  # Pause current playback
            print("Playback paused")
            return current_time, False  # Return new time and paused state
    except SpotifyException as e:
        print(f"Spotify error: {e}")  # Log any API errors (e.g., no active device)
        return last_play_pause_time, None  # No state change on error

# Volume adjustment function
def adjust_volume(sp, lm_list, last_volume_time, last_volume, img):
    """Adjust Spotify volume based on thumb-index distance."""
    current_time = time.time()  # Get current timestamp
    # Enforce cooldown to prevent excessive API calls
    if current_time - last_volume_time < VOLUME_UPDATE_INTERVAL:
        return last_volume_time, last_volume

    # Get thumb and index fingertip coordinates
    thumb_tip = np.array(lm_list[4][1:3])  # Thumb tip (landmark 4)
    index_tip = np.array(lm_list[8][1:3])  # Index tip (landmark 8)
    # Calculate Euclidean distance between thumb and index tips
    distance = np.linalg.norm(thumb_tip - index_tip)
    # Map distance (20-200 pixels) to volume level (0-100%)
    vol_level = int(np.interp(distance, [20, 200], [0, 100]))

    try:
        sp.volume(vol_level)  # Set Spotify volume
        print(f"Volume set to {vol_level}%")  # Log new volume
        # Draw visual feedback on image (circles, line, text)
        draw_volume_feedback(img, thumb_tip, index_tip, vol_level)
        return current_time, vol_level  # Return updated time and volume
    except SpotifyException as e:
        # Handle rate limit error (429) with a backoff
        if e.http_status == 429:
            print("Rate limit hit, backing off...")
            time.sleep(1)  # Wait 1 second before retrying
        else:
            print(f"Spotify error: {e}")  # Log other errors
        return last_volume_time, last_volume  # Return old values on failure

# Volume feedback drawing function
def draw_volume_feedback(img, thumb_tip, index_tip, vol_level):
    """Draw circles, line, and volume text on the image."""
    thumb_pos = tuple(thumb_tip.astype(int))  # Convert thumb coords to integer tuple
    index_pos = tuple(index_tip.astype(int))  # Convert index coords to integer tuple
    cv.circle(img, thumb_pos, 15, (255, 0, 0), cv.FILLED)  # Draw blue circle at thumb
    cv.circle(img, index_pos, 15, (255, 0, 0), cv.FILLED)  # Draw blue circle at index
    cv.line(img, thumb_pos, index_pos, (255, 255, 255), 2)  # Draw white line between
    # Display volume percentage in green text
    cv.putText(img, f"Volume: {int(vol_level)}%", (50, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Song skipping function
def handle_song_skip(sp, lm_list, skip_data, last_skip_time, img):
    """Skip tracks based on index finger movement with confirmation and feedback."""
    current_time = time.time()  # Get current timestamp
    # Enforce cooldown between skips
    if current_time - last_skip_time < SKIP_COOLDOWN:
        return skip_data, last_skip_time

    index_tip = np.array(lm_list[8][1:3])  # Index fingertip coordinates
    history, start_time = skip_data  # Unpack skip tracking data

    if start_time is None:
        start_time = current_time  # Start tracking if not already

    history.append(index_tip[0])  # Add current x-coordinate to history
    if len(history) > SKIP_FRAME_COUNT:
        history.pop(0)  # Keep history at fixed length

    # Draw yellow line to visualize index finger movement
    if len(history) > 1:
        for i in range(1, len(history)):
            cv.line(img, (int(history[i-1]), index_tip[1]), 
                    (int(history[i]), index_tip[1]), (0, 255, 255), 2)
        cv.putText(img, "Tracking skip...", (50, 130), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Wait for gesture confirmation duration
    if current_time - start_time < SKIP_CONFIRM_DURATION:
        return (history, start_time), last_skip_time

    # Calculate displacement for skip detection
    if len(history) == SKIP_FRAME_COUNT:
        avg_start = sum(history[:SKIP_FRAME_COUNT//2]) / (SKIP_FRAME_COUNT//2)  # Average of first half
        avg_end = sum(history[SKIP_FRAME_COUNT//2:]) / (SKIP_FRAME_COUNT//2)  # Average of second half
        displacement = avg_end - avg_start  # Movement direction and magnitude

        if displacement > SKIP_THRESHOLD:  # Right movement for next track
            try:
                sp.next_track()
                print("Next track")
                return ([], None), current_time  # Reset tracking after skip
            except SpotifyException as e:
                print(f"Spotify error: {e}")
        elif displacement < -SKIP_THRESHOLD:  # Left movement for previous track
            try:
                sp.previous_track()
                print("Previous track")
                return ([], None), current_time  # Reset tracking after skip
            except SpotifyException as e:
                print(f"Spotify error: {e}")

    return (history, start_time), last_skip_time  # Return current tracking state

# Toggle controls function
def toggle_controls(extended, controls_active, toggle_start_time):
    """Toggle gesture controls on/off with thumb and pinky gesture."""
    # Toggle gesture: thumb and pinky extended, others folded
    toggle_gesture = extended[0] and extended[4] and not any(extended[1:4])
    current_time = time.time()

    if toggle_gesture:
        if toggle_start_time is None:
            toggle_start_time = current_time  # Start timing toggle gesture
        elif current_time - toggle_start_time > TOGGLE_DURATION:
            controls_active = not controls_active  # Toggle control state
            print("Controls toggled to", "active" if controls_active else "inactive")
            toggle_start_time = None  # Reset timer after toggle
    else:
        toggle_start_time = None  # Reset if gesture not held

    return controls_active, toggle_start_time  # Return updated state and timer

# Display status function
def display_status(img, controls_active, toggle_start_time, fps, playback_state):
    """Display control status, playback state, and FPS on the image."""
    # Display control state in green (active) or red (inactive)
    status_text = "Controls: Active" if controls_active else "Controls: Inactive"
    status_color = (0, 255, 0) if controls_active else (0, 0, 255)
    cv.putText(img, status_text, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Display playback state in green (playing) or red (paused)
    playback_text = "Playing" if playback_state else "Paused"
    playback_color = (0, 255, 0) if playback_state else (255, 0, 0)
    cv.putText(img, playback_text, (50, 160), cv.FONT_HERSHEY_SIMPLEX, 1, playback_color, 2)

    # Show toggle feedback in yellow if gesture is being held 
    if toggle_start_time is not None:
        cv.putText(img, "Toggling controls...", (50, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display FPS in blue at top-left
    cv.putText(img, f'FPS: {int(fps)}', (40, 50),
               cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)

# Main function
def main():
    """Run the hand gesture Spotify control system."""
    sp = initialize_spotify()  # Initialize Spotify API client
    cap = initialize_camera()  # Set up camera
    detector = htm.handDetector(detectionCon=0.7)  # Initialize hand detector with 70% confidence (subject to change)

    # Initialize state variables
    controls_active = True  # Start with controls enabled
    toggle_start_time = None  # Timer for toggle gesture
    last_volume_time = 0  # Last time volume was updated
    last_volume = 0  # Last volume level set
    last_skip_time = 0  # Last time a skip occurred
    last_play_pause_time = 0  # Last time play/pause toggled
    playback_state = False  # Start with paused state
    skip_data = ([], None)  # (history, start_time) for skip tracking
    prev_time = 0  # Previous frame time for FPS calculation

    # Main loop: process frames and handle gestures
    while True:
        success, img = cap.read()  # Capture frame from camera
        if not success:
            break  # Exit if frame capture fails

        img, lm_list = detect_hand_landmarks(img, detector)  # Detect hand landmarks tips, mcps, etc
        current_time = time.time()  # Current timestamp
        # Calculate FPS based on frame time difference
        fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
        prev_time = current_time

        if lm_list:  # If hand landmarks are detected (points on fingers / palm etc)
            extended = get_extended_fingers(lm_list)  # Get extended finger states
            # Update control state based on toggle gesture
            controls_active, toggle_start_time = toggle_controls(extended, controls_active, toggle_start_time)

            if controls_active:  # Process gestures only if controls are active
                if all(extended):  # Volume gesture: all fingers extended
                    last_volume_time, last_volume = adjust_volume(sp, lm_list, last_volume_time, last_volume, img)
                    skip_data = ([], None)  # Reset skip tracking
                elif extended[1] and not extended[0] and not any(extended[2:]):  # Skip gesture: index only
                    skip_data, last_skip_time = handle_song_skip(sp, lm_list, skip_data, last_skip_time, img)
                elif extended[1] and extended[2] and not extended[0] and not any(extended[3:5]):  # Play/pause gesture
                    last_play_pause_time, new_state = play_pause(sp, lm_list, last_play_pause_time, img)
                    if new_state is not None:  # Update playback state if changed
                        playback_state = new_state
                    skip_data = ([], None)  # Reset skip tracking
                else:
                    skip_data = ([], None)  # Reset skip if no gesture matches

        # Update display with current states
        display_status(img, controls_active, toggle_start_time, fps, playback_state)
        cv.imshow("img", img)  # Show processed frame
        if cv.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break

    cap.release()  # Release camera resource
    cv.destroyAllWindows()  # Close all OpenCV windows

# Entry point: run the program
if __name__ == "__main__":
    main()
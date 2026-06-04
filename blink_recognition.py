import cv2
import mediapipe as mp
import time
import numpy as np
import sys

# Initialize MediaPipe Face Mesh using the new API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create FaceLandmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# Download the model file first
import urllib.request
import os

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

landmarker = FaceLandmarker.create_from_options(options)

# Eye landmarks indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(eye_points):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C)

# VIDEO SOURCE SELECTION
print("Blink Detection System")
print("Choose video source:")
print("1. Webcam (live)")
print("2. Video file")
choice = input("Enter choice (1 or 2): ").strip()

if choice == "2":
    video_path = input("Enter video file path: ").strip()
    # Remove quotes if user copied path with quotes
    video_path = video_path.strip('"').strip("'")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    print(f"Processing video: {video_path}")
else:
    cap = cv2.VideoCapture(0)
    print("Using webcam")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # For some webcams
    fps = 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
if total_frames > 0:
    print(f"Total frames: {total_frames}")
    print(f"Video duration: {total_frames/fps:.2f} seconds")

blink_count = 0
start_time = time.time()
eyes_closed = False
EAR_THRESHOLD = 0.2
frame_count = 0

print("\nProcessing... Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("\nEnd of video reached" if total_frames > 0 else "\nWebcam stopped")
        break
    
    frame_count += 1
    
    # Flip for mirror effect (only for webcam)
    if choice == "1":
        frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp in milliseconds
    timestamp_ms = int((frame_count / fps) * 1000)
    
    # Process the frame
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            h, w = frame.shape[:2]
            
            # Get eye points
            left_eye_points = [(face_landmarks[i].x * w, face_landmarks[i].y * h)
                              for i in LEFT_EYE]
            right_eye_points = [(face_landmarks[i].x * w, face_landmarks[i].y * h)
                               for i in RIGHT_EYE]
            
            # Calculate EAR
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            if avg_ear < EAR_THRESHOLD:
                if not eyes_closed:
                    blink_count += 1
                    eyes_closed = True
                    print(f"Blink #{blink_count} detected at {frame_count/fps:.2f}s")
                cv2.putText(frame, "BLINK!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                eyes_closed = False
            
            # Draw eye landmarks
            for point in left_eye_points + right_eye_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    elapsed = time.time() - start_time
    video_time = frame_count / fps  # Actual video time
    bpm = (blink_count / video_time) * 60 if video_time > 0 else 0
    
    # Display stats
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blinks/Min: {bpm:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show progress for video files
    if total_frames > 0:
        progress = (frame_count / total_frames) * 100
        cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    
    cv2.imshow('Blink Detection', frame)
    
    # Adjust wait time based on source
    wait_time = 1 if choice == "1" else int(1000 / fps)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

# Final statistics
print("\nFINAL STATISTICS")
print(f"Total Blinks: {blink_count}")
print(f"Video Duration: {video_time:.2f} seconds")
print(f"Processing Time: {elapsed:.2f} seconds")
print(f"Average Blinks/Minute: {bpm:.2f}")
print(f"Frames Processed: {frame_count}")
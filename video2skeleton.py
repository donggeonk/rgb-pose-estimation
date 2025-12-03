import cv2
import time
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Global variables for Tasks API callback
latest_result = None

def pose_detection():
    """
    Pose detection: capture, detect, draw skeleton, save 3D coordinates to NumPy
    """
    print("=== Pose Detection with 3D World Coordinates ===")
    
    # Import Tasks API components
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Check if model file exists
    model_path = 'model/pose_landmarker_full.task'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return False
    
    def result_callback(result, output_image: mp.Image, timestamp_ms: int):
        """Callback to store detection results"""
        global latest_result
        latest_result = result
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configure PoseLandmarker
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_poses=1,
        min_pose_detection_confidence=0.5, # lower to detect easier
        min_pose_presence_confidence=0.5, # lower to keep tracking longer
        min_tracking_confidence=0.5 # lower to track motion better
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return False
    
    # Initialize data storage
    pose_data_list = []  # List to collect pose data
    timestamps = []
    
    print("Pose detection started!")
    print("Press 'q' to quit and save data")
    
    frame_count = 0
    start_time = time.time()
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame")
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Get timestamp and process frame
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            display_frame = frame.copy()
            
            # Process detection results
            if latest_result and latest_result.pose_landmarks:
                # Get 2D landmarks for drawing
                landmarks = latest_result.pose_landmarks[0]
                h, w, _ = display_frame.shape
                
                # Draw landmark points
                for landmark in landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)
                
                # Draw skeleton connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
                    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
                    (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32)
                ]
                
                for start_idx, end_idx in connections:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_landmark = landmarks[start_idx]
                        end_landmark = landmarks[end_idx]
                        
                        start_x = int(start_landmark.x * w)
                        start_y = int(start_landmark.y * h)
                        end_x = int(end_landmark.x * w)
                        end_y = int(end_landmark.y * h)
                        
                        cv2.line(display_frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                
                # Save 3D world coordinates to NumPy array
                if latest_result.pose_world_landmarks:
                    world_landmarks = latest_result.pose_world_landmarks[0]
                    
                    # Extract all 33 landmarks (x, y, z) into a 33x3 array
                    landmarks_array = np.array([
                        [landmark.x, landmark.y, landmark.z] 
                        for landmark in world_landmarks
                    ])
                    
                    pose_data_list.append(landmarks_array)
                    timestamps.append(timestamp_ms)
                    
                    # Status update every 30 frames
                    # if frame_count % 30 == 0:
                    #     print(f"Frame {frame_count} | Poses recorded: {len(pose_data_list)} | FPS: {fps:.1f}")
            
            # Show FPS (only overlay)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Pose Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save data to NumPy files
    if pose_data_list:
        # Convert list to 3D NumPy array: (num_frames, 33_landmarks, 3_coordinates)
        pose_array = np.array(pose_data_list)
        timestamps_array = np.array(timestamps)
        
        # Generate filenames with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pose_filename = os.path.join(output_dir, f'realtime_pose_3d_coordinate_{timestamp_str}.npy')
        timestamps_filename = os.path.join(output_dir, f'realtime_pose_timestamps_{timestamp_str}.npy')
        metadata_filename = os.path.join(output_dir, f'realtime_pose_metadata_{timestamp_str}.txt')

        #  Save pose data
        np.save(pose_filename, pose_array)
        print(f"\n{'='*60}")
        print(f"Saved pose data: {pose_filename}")
        print(f"Shape: {pose_array.shape} (frames, landmarks, coordinates)")
        
        # Save timestamps
        np.save(timestamps_filename, timestamps_array)
        print(f"Saved timestamps: {timestamps_filename}")
        
        # Save metadata
        landmark_names = [
            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",
            "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
            "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
            "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
            "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
            "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
            "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX"
        ]
        
        with open(metadata_filename, 'w') as f:
            f.write(f"Pose Data Metadata\n")
            f.write(f"{'='*60}\n")
            f.write(f"Recording date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total frames: {len(pose_data_list)}\n")
            f.write(f"Array shape: {pose_array.shape}\n")
            f.write(f"Coordinates: 3D world coordinates in meters\n")
            f.write(f"Reference: Hip center (midpoint of landmarks 23 and 24)\n")
            f.write(f"\nLandmark indices:\n")
            for i, name in enumerate(landmark_names):
                f.write(f"  {i:2d}: {name}\n")
            f.write(f"\nUsage example:\n")
            f.write(f"  import numpy as np\n")
            f.write(f"  data = np.load('{pose_filename}')\n")
            f.write(f"  # Access frame 0, landmark 0 (NOSE), x-coordinate:\n")
            f.write(f"  nose_x = data[0, 0, 0]\n")
            f.write(f"  # Access all frames, left wrist (landmark 15):\n")
            f.write(f"  left_wrist = data[:, 15, :]\n")
        
        print(f"Saved metadata: {metadata_filename}")
        
        # Show example of how to load and use the data
        print(f"\nPose array shape: {pose_array.shape}")
        print(f"  - Dimension 0: {pose_array.shape[0]} frames")
        print(f"  - Dimension 1: {pose_array.shape[1]} landmarks")
        print(f"  - Dimension 2: {pose_array.shape[2]} coordinates (x, y, z)")
    else:
        print("No pose data recorded!")
    
    # Final summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    # print(f"\n{'='*60}")
    # print("SESSION SUMMARY")
    # print(f"{'='*60}")
    # print(f"Total frames processed: {frame_count}")
    # print(f"Poses recorded: {len(pose_data_list)}")
    # print(f"Total time: {total_time:.1f}s")
    print(f"\nAverage FPS: {avg_fps:.1f}")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    pose_detection()
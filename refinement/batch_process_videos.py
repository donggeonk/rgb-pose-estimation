import os
import glob
import numpy as np
from mediapipe_pose import PoseDetectionApp

def batch_process_rgb_videos(input_folder, output_folder):
    """
    Process all MP4 videos in input_folder and save MediaPipe 3D coordinates to output_folder.
    """
    # 1. Setup directories
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Find all video files
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {input_folder}")
        return

    print(f"Found {len(video_files)} videos to process.")

    # 3. Process each video
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]
        
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_name}")
        
        # Initialize your app with the video file
        # Note: We modify the collector inside the app to point to our specific output folder
        app = PoseDetectionApp(video_source=video_path)
        
        # Override the default output directory logic to keep things organized for training
        # We want: output_folder/video_id/pose_3d.npy
        target_dir = os.path.join(output_folder, video_id)
        app.collector.output_base_dir = output_folder 
        
        # Run the detection
        success = app.run()
        
        if success:
            print(f"Successfully processed {video_name}")
        else:
            print(f"Failed to process {video_name}")

if __name__ == "__main__":
    # Configuration
    RGB_VIDEO_DIR = "rgb_video"          # Folder containing your .mp4 files
    MEDIAPIPE_OUT_DIR = "mediapipe_output" # Folder to save the .npy files
    
    batch_process_rgb_videos(RGB_VIDEO_DIR, MEDIAPIPE_OUT_DIR)
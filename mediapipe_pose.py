import cv2
import time
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional, Union

# Global variables for Tasks API callback
latest_result = None


class PoseDetector:
    """Handles MediaPipe pose detection"""
    
    def __init__(self, model_path: str = 'model/pose_landmarker_full.task'):
        self.model_path = model_path
        self.landmarker = None
        self.latest_result = None
        
    def setup(self, running_mode: str = 'LIVE_STREAM') -> bool:
        """
        Initialize the pose landmarker
        
        Args:
            running_mode: 'LIVE_STREAM' for webcam, 'VIDEO' for video files
        """
        if not os.path.exists(self.model_path):
            print(f"Error: Model file '{self.model_path}' not found!")
            return False
        
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Set running mode
        if running_mode == 'LIVE_STREAM':
            mode = VisionRunningMode.LIVE_STREAM
            
            def result_callback(result, output_image: mp.Image, timestamp_ms: int):
                """Callback to store detection results"""
                global latest_result
                latest_result = result
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=mode,
                result_callback=result_callback,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:  # VIDEO mode
            mode = VisionRunningMode.VIDEO
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=mode,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.running_mode = running_mode
        return True
    
    def detect(self, frame: np.ndarray, timestamp_ms: int):
        """Process a frame and return detection results"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        if self.running_mode == 'LIVE_STREAM':
            self.landmarker.detect_async(mp_image, timestamp_ms)
            return latest_result
        else:  # VIDEO mode
            return self.landmarker.detect_for_video(mp_image, timestamp_ms)
    
    def close(self):
        """Clean up resources"""
        if self.landmarker:
            self.landmarker.close()


class VideoCapture:
    """Handles video capture from camera or video file"""
    
    def __init__(self, source: Union[int, str] = 0):
        """
        Initialize video capture
        
        Args:
            source: Camera ID (int) or video file path (str)
                   - 0: Default camera
                   - "path/to/video.mp4": Video file
        """
        self.source = source
        self.cap = None
        self.is_camera = isinstance(source, int)
        self.fps = 30.0  # Default FPS
        self.total_frames = 0
        
    def open(self) -> bool:
        """Open camera or video file"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            if self.is_camera:
                print(f"Error: Cannot open camera {self.source}")
            else:
                print(f"Error: Cannot open video file: {self.source}")
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30.0  # Default if FPS is not available
        
        # Print info
        if self.is_camera:
            print(f"Camera {self.source} opened successfully")
        else:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            print(f"Video file opened: {self.source}")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {self.fps:.1f}")
            print(f"  Total frames: {self.total_frames}")
            print(f"  Duration: {duration:.2f}s")
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from camera or video file"""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        """Release camera/video resources"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def get_fps(self) -> float:
        """Get video FPS"""
        return self.fps
    
    def get_frame_count(self) -> int:
        """Get total frame count (0 for camera)"""
        return self.total_frames
    
    def is_video_file(self) -> bool:
        """Check if source is a video file"""
        return not self.is_camera


class SkeletonRenderer:
    """Handles drawing skeleton on frames"""
    
    # Skeleton connections (MediaPipe pose connections)
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32)
    ]
    
    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks, color: Tuple[int, int, int] = (0, 255, 0)):
        """Draw landmark points on frame"""
        h, w, _ = frame.shape
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, color, -1)
    
    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks, color: Tuple[int, int, int] = (255, 0, 0)):
        """Draw skeleton connections on frame"""
        h, w, _ = frame.shape
        
        for start_idx, end_idx in SkeletonRenderer.CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                start_x = int(start_landmark.x * w)
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w)
                end_y = int(end_landmark.y * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    @staticmethod
    def draw_info(frame: np.ndarray, fps: float, frame_count: int, progress: Optional[float] = None):
        """Draw FPS, frame count, and optional progress on frame"""
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if progress is not None:
            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


class PoseDataCollector:
    """Collects and saves pose data"""
    
    LANDMARK_NAMES = [
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
    
    def __init__(self, output_base_dir: str = 'output'):
        self.output_base_dir = output_base_dir
        self.session_dir = None
        self.session_timestamp = None
        self.pose_data_list = []
        self.timestamps = []
        
    def create_session(self, source_name: Optional[str] = None) -> str:
        """
        Create a new session folder
        
        Args:
            source_name: Optional name to include in folder (e.g., video filename)
        """
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
        
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create folder name with optional source name
        if source_name:
            folder_name = f"{self.session_timestamp}_{source_name}"
        else:
            folder_name = self.session_timestamp
        
        self.session_dir = os.path.join(self.output_base_dir, folder_name)
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"Session folder: {self.session_dir}")
        return self.session_dir
    
    def add_pose(self, world_landmarks, timestamp_ms: int):
        """Add a pose to the collection"""
        landmarks_array = np.array([
            [landmark.x, landmark.y, landmark.z] 
            for landmark in world_landmarks
        ])
        self.pose_data_list.append(landmarks_array)
        self.timestamps.append(timestamp_ms)
    
    def save(self) -> bool:
        """Save collected data to files"""
        if not self.pose_data_list:
            print("No pose data recorded!")
            self._cleanup_empty_folder()
            return False
        
        # Convert to arrays
        pose_array = np.array(self.pose_data_list)
        timestamps_array = np.array(self.timestamps)
        
        # Save files
        self._save_pose_data(pose_array)
        self._save_timestamps(timestamps_array)
        self._save_metadata(pose_array)
        
        # Print summary
        self._print_summary(pose_array)
        return True
    
    def _save_pose_data(self, pose_array: np.ndarray):
        """Save pose coordinates"""
        filename = os.path.join(self.session_dir, 'pose_3d_coordinates.npy')
        np.save(filename, pose_array)
        print(f"Saved pose data: pose_3d_coordinates.npy")
        print(f"Shape: {pose_array.shape} (frames, landmarks, coordinates)")
    
    def _save_timestamps(self, timestamps_array: np.ndarray):
        """Save timestamps"""
        filename = os.path.join(self.session_dir, 'timestamps.npy')
        np.save(filename, timestamps_array)
        print(f"Saved timestamps: timestamps.npy")
    
    def _save_metadata(self, pose_array: np.ndarray):
        """Save metadata file"""
        filename = os.path.join(self.session_dir, 'metadata.txt')
        
        with open(filename, 'w') as f:
            f.write(f"Pose Data Metadata\n")
            f.write(f"{'='*60}\n")
            f.write(f"Session: {self.session_timestamp}\n")
            f.write(f"Recording date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total frames: {len(self.pose_data_list)}\n")
            f.write(f"Array shape: {pose_array.shape}\n")
            f.write(f"Coordinates: 3D world coordinates in meters\n")
            f.write(f"Reference: Hip center (midpoint of landmarks 23 and 24)\n")
            f.write(f"\nFiles in this session:\n")
            f.write(f"  - pose_3d_coordinates.npy: 3D pose data\n")
            f.write(f"  - timestamps.npy: Frame timestamps in milliseconds\n")
            f.write(f"  - metadata.txt: This file\n")
            f.write(f"\nLandmark indices:\n")
            for i, name in enumerate(self.LANDMARK_NAMES):
                f.write(f"  {i:2d}: {name}\n")
            f.write(f"\nUsage example:\n")
            f.write(f"  import numpy as np\n")
            f.write(f"  data = np.load('{self.session_dir}/pose_3d_coordinates.npy')\n")
            f.write(f"  timestamps = np.load('{self.session_dir}/timestamps.npy')\n")
            f.write(f"  # Access frame 0, landmark 0 (NOSE), x-coordinate:\n")
            f.write(f"  nose_x = data[0, 0, 0]\n")
            f.write(f"  # Access all frames, left wrist (landmark 15):\n")
            f.write(f"  left_wrist = data[:, 15, :]\n")
        
        print(f"Saved metadata: metadata.txt")
    
    def _print_summary(self, pose_array: np.ndarray):
        """Print data summary"""
        print(f"\n{'='*60}")
        print(f"Session saved to: {self.session_dir}")
        print(f"{'='*60}")
        print(f"Pose array shape: {pose_array.shape}")
        print(f"  - Dimension 0: {pose_array.shape[0]} frames")
        print(f"  - Dimension 1: {pose_array.shape[1]} landmarks")
        print(f"  - Dimension 2: {pose_array.shape[2]} coordinates (x, y, z)")
    
    def _cleanup_empty_folder(self):
        """Remove empty session folder"""
        if self.session_dir:
            try:
                os.rmdir(self.session_dir)
                print(f"Removed empty session folder: {self.session_dir}")
            except:
                pass


class PoseDetectionApp:
    """Main application orchestrator"""
    
    def __init__(self, video_source: Union[int, str] = 0):
        """
        Initialize app
        
        Args:
            video_source: Camera ID (int) or video file path (str)
        """
        self.detector = PoseDetector()
        self.camera = VideoCapture(video_source)
        self.renderer = SkeletonRenderer()
        self.collector = PoseDataCollector()
        self.video_source = video_source
        
    def run(self) -> bool:
        """Run the pose detection application"""
        print("=== Pose Detection with 3D World Coordinates ===")
        
        # Open video source
        if not self.camera.open():
            return False
        
        # Determine running mode based on source type
        is_video_file = self.camera.is_video_file()
        running_mode = 'VIDEO' if is_video_file else 'LIVE_STREAM'
        
        # Setup detector with appropriate mode
        if not self.detector.setup(running_mode):
            return False
        
        # Create session folder (include video name if processing file)
        if is_video_file:
            video_basename = os.path.splitext(os.path.basename(str(self.video_source)))[0]
            self.collector.create_session(video_basename)
        else:
            self.collector.create_session()
        
        # Get video properties
        total_frames = self.camera.get_frame_count()
        video_fps = self.camera.get_fps()
        
        # Main loop
        print("Pose detection started!")
        if is_video_file:
            print(f"Processing video file: {total_frames} frames")
            print("Press 'q' to abort")
        else:
            print("Press 'q' to quit and save data")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    if is_video_file:
                        print("\nEnd of video reached")
                    else:
                        print("Error: Failed to read frame")
                    break
                
                frame_count += 1
                
                # Calculate timestamp
                if is_video_file:
                    # For video files, use frame-based timestamp
                    timestamp_ms = int((frame_count / video_fps) * 1000)
                else:
                    # For camera, use real-time timestamp
                    timestamp_ms = int(time.time() * 1000)
                
                # Detect pose
                result = self.detector.detect(frame, timestamp_ms)
                
                # Calculate processing FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Render
                display_frame = frame.copy()
                
                if result and result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    
                    # Draw skeleton
                    self.renderer.draw_landmarks(display_frame, landmarks)
                    self.renderer.draw_skeleton(display_frame, landmarks)
                    
                    # Collect 3D data
                    if result.pose_world_landmarks:
                        world_landmarks = result.pose_world_landmarks[0]
                        self.collector.add_pose(world_landmarks, timestamp_ms)
                
                # Calculate progress for video files
                progress = None
                if is_video_file and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                
                # Draw info
                self.renderer.draw_info(display_frame, fps, len(self.collector.pose_data_list), progress)
                
                # Print progress for video files (every 30 frames)
                if is_video_file and frame_count % 30 == 0:
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"Processing FPS: {fps:.1f} | Poses: {len(self.collector.pose_data_list)}")
                
                # Display
                cv2.imshow('Pose Detection', display_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Cleanup
            self._cleanup(start_time, frame_count)
        
        return True
    
    def _cleanup(self, start_time: float, frame_count: int):
        """Clean up resources and save data"""
        self.camera.release()
        self.detector.close()
        
        # Save collected data
        self.collector.save()
        
        # Print final summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nAverage processing FPS: {avg_fps:.1f}")
        print(f"{'='*60}")


def main():
    """Entry point"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        # Try to convert to int (camera ID), otherwise use as file path
        try:
            video_source = int(video_source)
        except ValueError:
            # Keep as string (file path)
            if not os.path.exists(video_source):
                print(f"Error: Video file not found: {video_source}")
                sys.exit(1)
    else:
        video_source = 0  # Default: camera 0
    
    app = PoseDetectionApp(video_source)
    app.run()


if __name__ == "__main__":
    main()
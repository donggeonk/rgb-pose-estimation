import cv2
import time
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional, Union
from scipy.spatial.transform import Rotation as R

# Global variables for Tasks API callback
latest_result = None


class GlobalTrajectoryEstimator:
    """
    Estimates global trajectory and orientation from local pose data.
    
    MediaPipe centers the pose at the hips every frame, so we must
    reconstruct global movement using multiple estimation techniques:
    
    1. Torso Orientation: Calculated from shoulder/hip vectors (ACCURATE)
    2. Global Position: Estimated via foot contact + velocity integration (APPROXIMATE)
    
    For true global position, MoCap or Visual SLAM is required.
    This provides the best possible estimate from monocular video.
    """
    
    def __init__(self, fps: float = 30.0):
        # Global state
        self.global_position = np.array([0.0, 0.0, 0.0])
        self.global_velocity = np.array([0.0, 0.0, 0.0])
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Previous frame data
        self.prev_landmarks = None
        self.prev_rotation = np.eye(3)
        self.prev_timestamp_ms = None
        
        # Foot contact state
        self.left_foot_planted = False
        self.right_foot_planted = False
        self.prev_left_foot_pos = None
        self.prev_right_foot_pos = None
        
        # Smoothing filters (exponential moving average)
        self.velocity_smoothing = 0.3  # Lower = smoother but more lag
        self.position_smoothing = 0.1
        
        # Height estimation (for ground plane)
        self.ground_height_history = []
        self.ground_height_window = 30  # frames
        
        # Drift correction
        self.stationary_threshold = 0.005  # meters/frame
        self.stationary_frames = 0
        self.stationary_frames_threshold = 5
        
    def reset(self):
        """Reset estimator to initial state"""
        self.global_position = np.array([0.0, 0.0, 0.0])
        self.global_velocity = np.array([0.0, 0.0, 0.0])
        self.prev_landmarks = None
        self.prev_rotation = np.eye(3)
        self.prev_timestamp_ms = None
        self.prev_left_foot_pos = None
        self.prev_right_foot_pos = None
        self.ground_height_history = []
        self.stationary_frames = 0
        
    def _calculate_torso_orientation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate torso orientation as a 3x3 rotation matrix.
        
        Coordinate System:
        - X-axis: Left to Right (hip vector)
        - Y-axis: Down to Up (spine vector)  
        - Z-axis: Back to Front (forward direction)
        
        Args:
            landmarks: (33, 3) array of landmark positions
            
        Returns:
            rotation_matrix: (3, 3) orthonormal rotation matrix
        """
        # Extract key joints
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Hip center (root position in local frame)
        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        
        # X-axis: Left hip → Right hip (normalized)
        hip_vec = right_hip - left_hip
        x_axis = hip_vec / (np.linalg.norm(hip_vec) + 1e-8)
        
        # Y-axis: Hip center → Shoulder center (spine direction)
        spine_vec = shoulder_center - hip_center
        y_axis = spine_vec / (np.linalg.norm(spine_vec) + 1e-8)
        
        # Z-axis: Cross product (forward direction)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        
        # Re-orthogonalize Y-axis to ensure orthonormal basis
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        # Build rotation matrix (columns are basis vectors)
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Ensure it's a valid rotation matrix (det = 1)
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] *= -1
            
        return rotation_matrix
    
    def _estimate_ground_height(self, landmarks: np.ndarray) -> float:
        """
        Estimate ground plane height from foot positions.
        Uses the lowest foot point as ground reference.
        """
        # Foot landmarks
        left_heel = landmarks[29]
        right_heel = landmarks[30]
        left_toe = landmarks[31]
        right_toe = landmarks[32]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Find lowest Y value (most negative in MediaPipe coords = lowest)
        foot_heights = [
            left_heel[1], right_heel[1],
            left_toe[1], right_toe[1],
            left_ankle[1], right_ankle[1]
        ]
        current_ground = min(foot_heights)
        
        # Maintain rolling average for stability
        self.ground_height_history.append(current_ground)
        if len(self.ground_height_history) > self.ground_height_window:
            self.ground_height_history.pop(0)
            
        return np.median(self.ground_height_history)
    
    def _detect_foot_contact(self, landmarks: np.ndarray) -> Tuple[bool, bool, np.ndarray, np.ndarray]:
        """
        Detect which foot is in contact with ground (planted).
        A planted foot should have minimal velocity.
        
        Returns:
            left_planted: bool
            right_planted: bool
            left_foot_pos: (3,) position
            right_foot_pos: (3,) position
        """
        # Use heel as foot contact point
        left_foot = landmarks[29]  # LEFT_HEEL
        right_foot = landmarks[30]  # RIGHT_HEEL
        
        # Estimate ground height
        ground_height = self._estimate_ground_height(landmarks)
        
        # Height threshold for contact (within 3cm of ground)
        contact_threshold = 0.03
        
        left_near_ground = abs(left_foot[1] - ground_height) < contact_threshold
        right_near_ground = abs(right_foot[1] - ground_height) < contact_threshold
        
        # Velocity-based detection (if we have previous frame)
        left_stationary = True
        right_stationary = True
        
        if self.prev_left_foot_pos is not None:
            left_velocity = np.linalg.norm(left_foot - self.prev_left_foot_pos)
            left_stationary = left_velocity < self.stationary_threshold
            
        if self.prev_right_foot_pos is not None:
            right_velocity = np.linalg.norm(right_foot - self.prev_right_foot_pos)
            right_stationary = right_velocity < self.stationary_threshold
        
        # Foot is planted if near ground AND stationary
        left_planted = left_near_ground and left_stationary
        right_planted = right_near_ground and right_stationary
        
        return left_planted, right_planted, left_foot, right_foot
    
    def _estimate_velocity_from_feet(self, landmarks: np.ndarray, 
                                      rotation: np.ndarray) -> np.ndarray:
        """
        Estimate body velocity using foot contact method.
        
        Logic: When a foot is planted on the ground, the apparent movement
        of that foot in the local frame is actually the inverse of body movement.
        
        Returns:
            velocity: (3,) estimated velocity in global frame
        """
        if self.prev_landmarks is None:
            return np.array([0.0, 0.0, 0.0])
        
        # Detect foot contact
        left_planted, right_planted, left_foot, right_foot = \
            self._detect_foot_contact(landmarks)
        
        velocity = np.array([0.0, 0.0, 0.0])
        num_contacts = 0
        
        # Left foot planted → use its apparent motion (inverted)
        if left_planted and self.prev_left_foot_pos is not None:
            left_delta = left_foot - self.prev_left_foot_pos
            # Invert: if foot appears to move backward, body moved forward
            velocity -= left_delta / self.dt
            num_contacts += 1
            
        # Right foot planted
        if right_planted and self.prev_right_foot_pos is not None:
            right_delta = right_foot - self.prev_right_foot_pos
            velocity -= right_delta / self.dt
            num_contacts += 1
        
        # Average if both feet planted
        if num_contacts > 0:
            velocity /= num_contacts
            
        # Transform to global frame using current rotation
        global_velocity = rotation @ velocity
        
        # Zero out vertical component (we're on ground plane)
        # Keep only horizontal movement (X and Z in typical world coords)
        global_velocity[1] = 0.0
        
        # Update foot positions for next frame
        self.prev_left_foot_pos = left_foot.copy()
        self.prev_right_foot_pos = right_foot.copy()
        
        return global_velocity
    
    def _estimate_velocity_from_hip(self, landmarks: np.ndarray,
                                     rotation: np.ndarray) -> np.ndarray:
        """
        Alternative velocity estimation using hip center movement.
        Less accurate but works when feet are not visible.
        
        Note: MediaPipe recenters hips, so this captures pose drift/errors,
        not true movement. Used as fallback only.
        """
        if self.prev_landmarks is None:
            return np.array([0.0, 0.0, 0.0])
        
        # Current and previous hip centers
        curr_hip = (landmarks[23] + landmarks[24]) / 2.0
        prev_hip = (self.prev_landmarks[23] + self.prev_landmarks[24]) / 2.0
        
        # Hip movement (in local frame)
        hip_delta = curr_hip - prev_hip
        
        # This is mostly noise since MediaPipe recenters, but can indicate drift
        velocity = hip_delta / self.dt
        
        # Transform to global
        global_velocity = rotation @ velocity
        
        # Heavily dampen since this is unreliable
        return global_velocity * 0.1
    
    def _apply_drift_correction(self, velocity: np.ndarray, 
                                 landmarks: np.ndarray) -> np.ndarray:
        """
        Apply drift correction when person appears stationary.
        
        If both feet are planted and velocity is very low for several frames,
        we assume the person is standing still and zero out velocity.
        """
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Check if both feet are planted
        left_planted, right_planted, _, _ = self._detect_foot_contact(landmarks)
        
        if left_planted and right_planted and velocity_magnitude < 0.1:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0
            
        # If stationary for enough frames, zero velocity
        if self.stationary_frames > self.stationary_frames_threshold:
            return np.array([0.0, 0.0, 0.0])
            
        return velocity
    
    def _smooth_velocity(self, new_velocity: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to velocity"""
        alpha = self.velocity_smoothing
        self.global_velocity = alpha * new_velocity + (1 - alpha) * self.global_velocity
        return self.global_velocity
    
    def process_frame(self, landmarks_array: np.ndarray, 
                      timestamp_ms: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Process a single frame and estimate global position/orientation.
        
        Args:
            landmarks_array: (33, 3) array of local coordinates from MediaPipe
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            global_position: (3,) estimated global position [x, y, z]
            rotation_matrix: (3, 3) torso orientation matrix
            global_velocity: (3,) estimated velocity [vx, vy, vz]
            debug_info: dict with additional information
        """
        # Calculate time delta
        if self.prev_timestamp_ms is not None:
            self.dt = (timestamp_ms - self.prev_timestamp_ms) / 1000.0
            if self.dt <= 0 or self.dt > 1.0:  # Invalid dt
                self.dt = 1.0 / self.fps
        
        # 1. Calculate torso orientation (this is accurate)
        rotation_matrix = self._calculate_torso_orientation(landmarks_array)
        
        # 2. Estimate velocity from foot contact
        foot_velocity = self._estimate_velocity_from_feet(landmarks_array, rotation_matrix)
        
        # 3. Fallback: hip-based velocity (if feet unreliable)
        hip_velocity = self._estimate_velocity_from_hip(landmarks_array, rotation_matrix)
        
        # 4. Combine velocities (prefer foot-based)
        left_planted, right_planted, _, _ = self._detect_foot_contact(landmarks_array)
        
        if left_planted or right_planted:
            raw_velocity = foot_velocity
        else:
            # No foot contact - use hip with heavy damping
            raw_velocity = hip_velocity
        
        # 5. Apply drift correction
        corrected_velocity = self._apply_drift_correction(raw_velocity, landmarks_array)
        
        # 6. Smooth velocity
        smoothed_velocity = self._smooth_velocity(corrected_velocity)
        
        # 7. Integrate position
        self.global_position = self.global_position + smoothed_velocity * self.dt
        
        # 8. Calculate hip center (local frame origin)
        hip_center = (landmarks_array[23] + landmarks_array[24]) / 2.0
        
        # 9. Store for next frame
        self.prev_landmarks = landmarks_array.copy()
        self.prev_rotation = rotation_matrix.copy()
        self.prev_timestamp_ms = timestamp_ms
        
        # Debug info
        debug_info = {
            'left_foot_planted': left_planted,
            'right_foot_planted': right_planted,
            'raw_velocity': raw_velocity.copy(),
            'smoothed_velocity': smoothed_velocity.copy(),
            'stationary_frames': self.stationary_frames,
            'hip_center_local': hip_center.copy(),
            'dt': self.dt
        }
        
        return self.global_position.copy(), rotation_matrix, smoothed_velocity.copy(), debug_info
    
    def get_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion [w, x, y, z]
        
        Useful for RL frameworks that prefer quaternion representation.
        """
        try:
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()  # Returns [x, y, z, w]
            # Reorder to [w, x, y, z] (common convention)
            return np.array([quat[3], quat[0], quat[1], quat[2]])
        except:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    def get_euler_angles(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles [roll, pitch, yaw] in radians.
        
        Uses XYZ convention (roll around X, pitch around Y, yaw around Z).
        """
        try:
            r = R.from_matrix(rotation_matrix)
            return r.as_euler('xyz', degrees=False)
        except:
            return np.array([0.0, 0.0, 0.0])


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
        self.source = source
        self.cap = None
        self.is_camera = isinstance(source, int)
        self.fps = 30.0
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
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30.0
        
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
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def get_fps(self) -> float:
        return self.fps
    
    def get_frame_count(self) -> int:
        return self.total_frames
    
    def is_video_file(self) -> bool:
        return not self.is_camera


class SkeletonRenderer:
    """Handles drawing skeleton on frames"""
    
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32)
    ]
    
    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks, color: Tuple[int, int, int] = (0, 255, 0)):
        h, w, _ = frame.shape
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, color, -1)
    
    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks, color: Tuple[int, int, int] = (255, 0, 0)):
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
    def draw_info(frame: np.ndarray, fps: float, frame_count: int, 
                  progress: Optional[float] = None,
                  global_pos: Optional[np.ndarray] = None,
                  euler_angles: Optional[np.ndarray] = None):
        """Draw FPS, frame count, and global position info"""
        y_offset = 30
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Frames: {frame_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30
        
        if progress is not None:
            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30
        
        if global_pos is not None:
            cv2.putText(frame, f"Pos: [{global_pos[0]:.2f}, {global_pos[1]:.2f}, {global_pos[2]:.2f}]", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 25
            
        if euler_angles is not None:
            # Convert to degrees for display
            angles_deg = np.degrees(euler_angles)
            cv2.putText(frame, f"Rot: [{angles_deg[0]:.1f}, {angles_deg[1]:.1f}, {angles_deg[2]:.1f}]", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


class PoseDataCollector:
    """Collects and saves pose data with global trajectory estimation"""
    
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
    
    def __init__(self, output_base_dir: str = 'output', fps: float = 30.0):
        self.output_base_dir = output_base_dir
        self.session_dir = None
        self.session_timestamp = None
        self.fps = fps
        
        # Data storage
        self.pose_data_list = []           # Local coordinates (33, 3)
        self.timestamps = []                # Timestamps (ms)
        self.global_positions = []          # Global position (3,)
        self.global_velocities = []         # Global velocity (3,)
        self.torso_rotations = []           # Rotation matrices (3, 3)
        self.torso_quaternions = []         # Quaternions (4,) [w, x, y, z]
        self.torso_euler_angles = []        # Euler angles (3,) [roll, pitch, yaw]
        
        # Global trajectory estimator
        self.estimator = GlobalTrajectoryEstimator(fps=fps)
        
    def set_fps(self, fps: float):
        """Update FPS for trajectory estimator"""
        self.fps = fps
        self.estimator = GlobalTrajectoryEstimator(fps=fps)
        
    def create_session(self, source_name: Optional[str] = None) -> str:
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
        
        self.session_timestamp = "global_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if source_name:
            folder_name = f"{self.session_timestamp}_{source_name}"
        else:
            folder_name = self.session_timestamp
        
        self.session_dir = os.path.join(self.output_base_dir, folder_name)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Reset estimator for new session
        self.estimator.reset()
        
        print(f"Session folder: {self.session_dir}")
        return self.session_dir
    
    def add_pose(self, world_landmarks, timestamp_ms: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add a pose to the collection and calculate global features.
        
        Returns:
            global_position: (3,) for visualization
            euler_angles: (3,) for visualization
        """
        # Convert landmarks to numpy array
        landmarks_array = np.array([
            [landmark.x, landmark.y, landmark.z] 
            for landmark in world_landmarks
        ])
        
        # Calculate global features
        global_pos, rotation_matrix, velocity, debug_info = \
            self.estimator.process_frame(landmarks_array, timestamp_ms)
        
        # Get quaternion and euler representations
        quaternion = self.estimator.get_quaternion(rotation_matrix)
        euler_angles = self.estimator.get_euler_angles(rotation_matrix)
        
        # Store all data
        self.pose_data_list.append(landmarks_array)
        self.timestamps.append(timestamp_ms)
        self.global_positions.append(global_pos)
        self.global_velocities.append(velocity)
        self.torso_rotations.append(rotation_matrix)
        self.torso_quaternions.append(quaternion)
        self.torso_euler_angles.append(euler_angles)
        
        return global_pos, euler_angles
    
    def save(self) -> bool:
        """Save collected data to files"""
        if not self.pose_data_list:
            print("No pose data recorded!")
            self._cleanup_empty_folder()
            return False
        
        # Convert to arrays
        pose_array = np.array(self.pose_data_list)
        timestamps_array = np.array(self.timestamps)
        global_pos_array = np.array(self.global_positions)
        global_vel_array = np.array(self.global_velocities)
        rotation_array = np.array(self.torso_rotations)
        quaternion_array = np.array(self.torso_quaternions)
        euler_array = np.array(self.torso_euler_angles)
        
        # Save all files
        self._save_file('pose_3d_coordinates.npy', pose_array, 
                       f"Shape: {pose_array.shape} (frames, landmarks, xyz)")
        self._save_file('timestamps.npy', timestamps_array,
                       f"Shape: {timestamps_array.shape} (frames,)")
        self._save_file('global_position.npy', global_pos_array,
                       f"Shape: {global_pos_array.shape} (frames, xyz)")
        self._save_file('global_velocity.npy', global_vel_array,
                       f"Shape: {global_vel_array.shape} (frames, xyz)")
        self._save_file('torso_rotation.npy', rotation_array,
                       f"Shape: {rotation_array.shape} (frames, 3, 3)")
        self._save_file('torso_quaternion.npy', quaternion_array,
                       f"Shape: {quaternion_array.shape} (frames, wxyz)")
        self._save_file('torso_euler.npy', euler_array,
                       f"Shape: {euler_array.shape} (frames, roll/pitch/yaw)")
        
        self._save_metadata(pose_array)
        self._print_summary(pose_array, global_pos_array)
        
        return True
    
    def _save_file(self, filename: str, data: np.ndarray, description: str):
        """Save a numpy file with description"""
        filepath = os.path.join(self.session_dir, filename)
        np.save(filepath, data)
        print(f"Saved {filename}: {description}")
    
    def _save_metadata(self, pose_array: np.ndarray):
        """Save comprehensive metadata file"""
        filename = os.path.join(self.session_dir, 'metadata.txt')
        
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("POSE DATA METADATA - Global Trajectory Estimation\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Session: {self.session_timestamp}\n")
            f.write(f"Recording date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total frames: {len(self.pose_data_list)}\n")
            f.write(f"FPS: {self.fps:.1f}\n")
            f.write(f"Duration: {len(self.pose_data_list) / self.fps:.2f} seconds\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("1. pose_3d_coordinates.npy\n")
            f.write("   - Shape: (frames, 33, 3)\n")
            f.write("   - Content: Local 3D coordinates relative to hip center\n")
            f.write("   - Units: Meters\n\n")
            
            f.write("2. timestamps.npy\n")
            f.write("   - Shape: (frames,)\n")
            f.write("   - Content: Frame timestamps\n")
            f.write("   - Units: Milliseconds\n\n")
            
            f.write("3. global_position.npy\n")
            f.write("   - Shape: (frames, 3)\n")
            f.write("   - Content: Estimated global position [x, y, z]\n")
            f.write("   - Units: Meters\n")
            f.write("   - Note: Estimated via foot contact dead reckoning\n\n")
            
            f.write("4. global_velocity.npy\n")
            f.write("   - Shape: (frames, 3)\n")
            f.write("   - Content: Estimated global velocity [vx, vy, vz]\n")
            f.write("   - Units: Meters/second\n\n")
            
            f.write("5. torso_rotation.npy\n")
            f.write("   - Shape: (frames, 3, 3)\n")
            f.write("   - Content: Torso orientation as rotation matrix\n")
            f.write("   - Convention: Body frame to world frame\n\n")
            
            f.write("6. torso_quaternion.npy\n")
            f.write("   - Shape: (frames, 4)\n")
            f.write("   - Content: Torso orientation as quaternion [w, x, y, z]\n")
            f.write("   - Use for: RL frameworks preferring quaternions\n\n")
            
            f.write("7. torso_euler.npy\n")
            f.write("   - Shape: (frames, 3)\n")
            f.write("   - Content: Euler angles [roll, pitch, yaw]\n")
            f.write("   - Units: Radians (XYZ convention)\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("COORDINATE SYSTEM\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("Body Frame (Torso):\n")
            f.write("  X-axis: Left hip → Right hip (lateral)\n")
            f.write("  Y-axis: Hip center → Shoulder center (vertical/spine)\n")
            f.write("  Z-axis: Back → Front (forward facing direction)\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("IMPORTANT NOTES FOR RL TRAINING\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("1. Global Position Accuracy:\n")
            f.write("   - Orientation (rotation/quaternion): ACCURATE\n")
            f.write("   - Position (global_position): APPROXIMATE (will drift)\n")
            f.write("   - For ground truth position, use MoCap data\n\n")
            
            f.write("2. Recommended Usage:\n")
            f.write("   - Use torso_rotation or torso_quaternion for orientation\n")
            f.write("   - Use global_velocity for motion estimation\n")
            f.write("   - Use global_position as rough trajectory reference\n")
            f.write("   - Train ST-GCN refinement model against MoCap ground truth\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("LANDMARK INDICES\n")
            f.write("-" * 70 + "\n\n")
            
            for i, name in enumerate(self.LANDMARK_NAMES):
                f.write(f"  {i:2d}: {name}\n")
            
            f.write("\n")
            f.write("-" * 70 + "\n")
            f.write("USAGE EXAMPLE\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("import numpy as np\n\n")
            f.write(f"# Load data\n")
            f.write(f"poses = np.load('{self.session_dir}/pose_3d_coordinates.npy')\n")
            f.write(f"positions = np.load('{self.session_dir}/global_position.npy')\n")
            f.write(f"rotations = np.load('{self.session_dir}/torso_rotation.npy')\n")
            f.write(f"quaternions = np.load('{self.session_dir}/torso_quaternion.npy')\n\n")
            f.write(f"# Frame 0 data\n")
            f.write(f"pose = poses[0]           # (33, 3) local joint positions\n")
            f.write(f"pos = positions[0]        # (3,) global position\n")
            f.write(f"rot = rotations[0]        # (3, 3) rotation matrix\n")
            f.write(f"quat = quaternions[0]     # (4,) quaternion [w,x,y,z]\n\n")
            f.write(f"# Transform local pose to global frame\n")
            f.write(f"global_pose = (rot @ pose.T).T + pos\n")
        
        print(f"Saved metadata: metadata.txt")
    
    def _print_summary(self, pose_array: np.ndarray, global_pos_array: np.ndarray):
        """Print data summary"""
        print(f"\n{'='*60}")
        print(f"Session saved to: {self.session_dir}")
        print(f"{'='*60}")
        print(f"\nLocal Pose Data:")
        print(f"  Shape: {pose_array.shape}")
        print(f"  - {pose_array.shape[0]} frames")
        print(f"  - {pose_array.shape[1]} landmarks")
        print(f"  - {pose_array.shape[2]} coordinates (x, y, z)")
        
        print(f"\nGlobal Trajectory:")
        print(f"  Total displacement: {np.linalg.norm(global_pos_array[-1] - global_pos_array[0]):.3f} m")
        print(f"  Max X: {global_pos_array[:, 0].max():.3f} m, Min X: {global_pos_array[:, 0].min():.3f} m")
        print(f"  Max Z: {global_pos_array[:, 2].max():.3f} m, Min Z: {global_pos_array[:, 2].min():.3f} m")
    
    def _cleanup_empty_folder(self):
        if self.session_dir:
            try:
                os.rmdir(self.session_dir)
                print(f"Removed empty session folder: {self.session_dir}")
            except:
                pass


class PoseDetectionApp:
    """Main application orchestrator"""
    
    def __init__(self, video_source: Union[int, str] = 0):
        self.detector = PoseDetector()
        self.camera = VideoCapture(video_source)
        self.renderer = SkeletonRenderer()
        self.collector = PoseDataCollector()
        self.video_source = video_source
        
    def run(self) -> bool:
        """Run the pose detection application"""
        print("=" * 60)
        print("Pose Detection with Global Trajectory Estimation")
        print("=" * 60)
        
        # Open video source
        if not self.camera.open():
            return False
        
        # Update collector with actual FPS
        self.collector.set_fps(self.camera.get_fps())
        
        # Determine running mode based on source type
        is_video_file = self.camera.is_video_file()
        running_mode = 'VIDEO' if is_video_file else 'LIVE_STREAM'
        
        # Setup detector with appropriate mode
        if not self.detector.setup(running_mode):
            return False
        
        # Create session folder
        if is_video_file:
            video_basename = os.path.splitext(os.path.basename(str(self.video_source)))[0]
            self.collector.create_session(video_basename)
        else:
            self.collector.create_session()
        
        # Get video properties
        total_frames = self.camera.get_frame_count()
        video_fps = self.camera.get_fps()
        
        # Main loop info
        print("\nPose detection started!")
        if is_video_file:
            print(f"Processing video file: {total_frames} frames at {video_fps:.1f} FPS")
            print("Press 'q' to abort")
        else:
            print("Press 'q' to quit and save data")
        print()
        
        frame_count = 0
        start_time = time.time()
        current_global_pos = None
        current_euler = None
        
        try:
            while True:
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
                    timestamp_ms = int((frame_count / video_fps) * 1000)
                else:
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
                    
                    # Collect 3D data with global estimation
                    if result.pose_world_landmarks:
                        world_landmarks = result.pose_world_landmarks[0]
                        current_global_pos, current_euler = \
                            self.collector.add_pose(world_landmarks, timestamp_ms)
                
                # Calculate progress for video files
                progress = None
                if is_video_file and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                
                # Draw info with global position
                self.renderer.draw_info(
                    display_frame, fps, len(self.collector.pose_data_list),
                    progress=progress,
                    global_pos=current_global_pos,
                    euler_angles=current_euler
                )
                
                # Print progress for video files
                if is_video_file and frame_count % 60 == 0:
                    pos_str = f"[{current_global_pos[0]:.2f}, {current_global_pos[2]:.2f}]" \
                              if current_global_pos is not None else "N/A"
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {fps:.1f} | Pos: {pos_str}")
                
                # Display
                cv2.imshow('Pose Detection - Global Trajectory', display_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
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
        print("=" * 60)


def main():
    """Entry point"""
    import sys
    
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        try:
            video_source = int(video_source)
        except ValueError:
            if not os.path.exists(video_source):
                print(f"Error: Video file not found: {video_source}")
                sys.exit(1)
    else:
        video_source = 0
    
    app = PoseDetectionApp(video_source)
    app.run()


if __name__ == "__main__":
    main()
import os
import sys
import cv2
import numpy as np

# --- CONFIGURATION ---
URDF_PATH = "unitree_ros/robots/g1_description/g1_29dof.urdf"
MODEL_PATH = "model/pose_landmarker_full.task"

# MediaPipe landmark indices
MP_LANDMARKS = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}


def check_pinocchio():
    """Check pinocchio installation and return the correct module"""
    try:
        import pinocchio as pin
        
        # Check which function is available
        if hasattr(pin, 'buildModelFromUrdf'):
            return pin, 'buildModelFromUrdf'
        elif hasattr(pin, 'buildModelFromXML'):
            return pin, 'buildModelFromXML'
        else:
            # List available functions for debugging
            urdf_funcs = [f for f in dir(pin) if 'urdf' in f.lower() or 'model' in f.lower()]
            print(f"Available pinocchio functions: {urdf_funcs}")
            
            # Try alternative import
            try:
                from pinocchio import buildModelFromUrdf
                return pin, 'direct_import'
            except ImportError:
                pass
            
            return pin, None
            
    except ImportError as e:
        print(f"Pinocchio not installed: {e}")
        print("\nInstall with: pip install pin")
        return None, None


def load_urdf_model(urdf_path: str, floating_base: bool = True):
    """Load URDF model with correct pinocchio API"""
    try:
        import pinocchio as pin
        
        # Method 1: Standard API (pinocchio >= 2.6)
        if hasattr(pin, 'buildModelFromUrdf'):
            if floating_base:
                model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
            else:
                model = pin.buildModelFromUrdf(urdf_path)
            return model
        
        # Method 2: Using urdf submodule
        if hasattr(pin, 'urdf'):
            if floating_base:
                model = pin.urdf.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
            else:
                model = pin.urdf.buildModelFromUrdf(urdf_path)
            return model
        
        # Method 3: RobotWrapper (older API)
        if hasattr(pin, 'RobotWrapper'):
            robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
            return robot.model
        
        # Method 4: Direct parser
        try:
            from pinocchio.robot_wrapper import RobotWrapper
            robot = RobotWrapper.BuildFromURDF(urdf_path)
            return robot.model
        except ImportError:
            pass
        
        raise RuntimeError("Could not find a working method to load URDF")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load URDF: {e}")


class G1Visualizer:
    """Visualizes G1 robot using Meshcat"""
    
    def __init__(self, model, data, urdf_path: str):
        import pinocchio as pin
        self.pin = pin
        self.model = model
        self.data = data
        self.urdf_path = urdf_path
        self.viz = None
        
    def setup(self) -> bool:
        """Initialize Meshcat visualizer"""
        try:
            from pinocchio.visualize import MeshcatVisualizer
            
            urdf_dir = os.path.dirname(os.path.abspath(self.urdf_path))
            
            # Try to load geometry models
            try:
                collision_model = self.pin.GeometryModel()
                visual_model = self.pin.GeometryModel()
                
                # Try different methods to build geometry
                if hasattr(self.pin, 'buildGeomFromUrdf'):
                    collision_model = self.pin.buildGeomFromUrdf(
                        self.model, self.urdf_path,
                        self.pin.GeometryType.COLLISION,
                        package_dirs=[urdf_dir, os.path.dirname(urdf_dir)]
                    )
                    visual_model = self.pin.buildGeomFromUrdf(
                        self.model, self.urdf_path,
                        self.pin.GeometryType.VISUAL,
                        package_dirs=[urdf_dir, os.path.dirname(urdf_dir)]
                    )
                elif hasattr(self.pin, 'urdf') and hasattr(self.pin.urdf, 'buildGeomFromUrdf'):
                    collision_model = self.pin.urdf.buildGeomFromUrdf(
                        self.model, self.urdf_path,
                        self.pin.GeometryType.COLLISION,
                        [urdf_dir]
                    )
                    visual_model = self.pin.urdf.buildGeomFromUrdf(
                        self.model, self.urdf_path,
                        self.pin.GeometryType.VISUAL,
                        [urdf_dir]
                    )
            except Exception as e:
                print(f"Could not load geometry: {e}")
                collision_model = self.pin.GeometryModel()
                visual_model = self.pin.GeometryModel()
            
            self.viz = MeshcatVisualizer(self.model, collision_model, visual_model)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel(color=[0.3, 0.6, 1.0, 0.8])
            
            print("\n✓ Meshcat visualizer started!")
            print("  Open browser at: http://127.0.0.1:7000/static/")
            return True
            
        except ImportError as e:
            print(f"\n⚠ Meshcat not available: {e}")
            print("  Install with: pip install meshcat")
            return False
        except Exception as e:
            print(f"\n⚠ Failed to setup visualizer: {e}")
            return False
    
    def display(self, q: np.ndarray):
        """Display robot configuration"""
        if self.viz is not None:
            self.viz.display(q)
    
    def add_target_markers(self, targets: dict):
        """Add spheres at target positions"""
        if self.viz is None:
            return
            
        try:
            import meshcat.geometry as g
            import meshcat.transformations as tf
            
            colors = {
                "L_hand": [1, 0, 0],
                "R_hand": [0, 1, 0],
                "L_foot": [0, 0, 1],
                "R_foot": [1, 1, 0],
            }
            
            for name, pos in targets.items():
                color = colors.get(name, [1, 1, 1])
                sphere = g.Sphere(0.03)
                material = g.MeshLambertMaterial(
                    color=int(color[0]*255)*65536 + int(color[1]*255)*256 + int(color[2]*255),
                    opacity=0.7
                )
                self.viz.viewer[f"target_{name}"].set_object(sphere, material)
                self.viz.viewer[f"target_{name}"].set_transform(tf.translation_matrix(pos))
        except Exception as e:
            print(f"Could not add markers: {e}")


class G1Retargeter:
    """Retargets human pose to Unitree G1 robot using Pinocchio IK"""
    
    def __init__(self, urdf_path: str):
        """Load robot model"""
        import pinocchio as pin
        self.pin = pin
        self.urdf_path = urdf_path
        
        # Load model using the helper function
        print("Loading G1 URDF...")
        try:
            self.model = load_urdf_model(urdf_path, floating_base=True)
            print("✓ Loaded model with floating base")
        except Exception as e:
            print(f"Could not load with floating base: {e}")
            try:
                self.model = load_urdf_model(urdf_path, floating_base=False)
                print("✓ Loaded model without floating base")
            except Exception as e2:
                raise RuntimeError(f"Failed to load URDF: {e2}")
        
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        
        # Get frame names
        self.frame_names = [f.name for f in self.model.frames]
        
        # Print model info
        print(f"\n{'='*50}")
        print("G1 ROBOT MODEL")
        print(f"{'='*50}")
        print(f"Joints: {self.model.njoints}, DOF: {self.model.nq}")
        print(f"Frames: {len(self.frame_names)}")
        
        # Auto-detect end-effector frames
        self.G1_FRAMES = self._auto_detect_frames()
        print(f"\nEnd-effector frames:")
        for k, v in self.G1_FRAMES.items():
            print(f"  {k}: {v}")
        print(f"{'='*50}\n")
        
        # Setup visualizer
        self.visualizer = G1Visualizer(self.model, self.data, urdf_path)
        
    def _auto_detect_frames(self) -> dict:
        """Auto-detect frame names for hands and feet"""
        frames = {}
        
        patterns = {
            "L_hand": ["left_wrist_roll_link", "left_wrist_yaw_link", "left_wrist_link",
                       "left_hand_link", "left_wrist_roll_rubber_hand", "left_zero_link"],
            "R_hand": ["right_wrist_roll_link", "right_wrist_yaw_link", "right_wrist_link",
                       "right_hand_link", "right_wrist_roll_rubber_hand", "right_zero_link"],
            "L_foot": ["left_ankle_roll_link", "left_ankle_link", "left_foot_link"],
            "R_foot": ["right_ankle_roll_link", "right_ankle_link", "right_foot_link"],
        }
        
        for key, search_terms in patterns.items():
            for term in search_terms:
                for frame_name in self.frame_names:
                    if term.lower() in frame_name.lower():
                        frames[key] = frame_name
                        break
                if key in frames:
                    break
            
            if key not in frames:
                # Fallback: search for partial matches
                keyword = "wrist" if "hand" in key else "ankle"
                side = "left" if key.startswith("L") else "right"
                for frame_name in self.frame_names:
                    if side in frame_name.lower() and keyword in frame_name.lower():
                        frames[key] = frame_name
                        break
                
                if key not in frames:
                    print(f"WARNING: Could not find frame for {key}")
        
        return frames
    
    def get_frame_id(self, frame_name: str) -> int:
        """Get frame ID by name"""
        if frame_name not in self.frame_names:
            return -1
        return self.model.getFrameId(frame_name)
    
    def retarget(self, landmarks_3d: np.ndarray, max_iterations: int = 200, 
                 visualize: bool = True) -> np.ndarray:
        """Retarget human pose to robot using IK"""
        pin = self.pin
        
        # Get valid frames
        valid_frames = {k: v for k, v in self.G1_FRAMES.items() 
                       if self.get_frame_id(v) >= 0}
        
        if not valid_frames:
            print("ERROR: No valid frames found!")
            return self.q
        
        # Extract targets
        targets = {}
        if "L_hand" in valid_frames:
            targets["L_hand"] = landmarks_3d[MP_LANDMARKS["LEFT_WRIST"]]
        if "R_hand" in valid_frames:
            targets["R_hand"] = landmarks_3d[MP_LANDMARKS["RIGHT_WRIST"]]
        if "L_foot" in valid_frames:
            targets["L_foot"] = landmarks_3d[MP_LANDMARKS["LEFT_ANKLE"]]
        if "R_foot" in valid_frames:
            targets["R_foot"] = landmarks_3d[MP_LANDMARKS["RIGHT_ANKLE"]]
        
        # Scaling
        human_height = self._estimate_height(landmarks_3d)
        g1_height = 1.27
        scale = g1_height / human_height if human_height > 0.1 else 1.0
        
        print(f"\nScaling: human={human_height:.2f}m → G1={g1_height:.2f}m (scale={scale:.2f})")
        
        # Center on hip
        hip_center = (landmarks_3d[MP_LANDMARKS["LEFT_HIP"]] + 
                      landmarks_3d[MP_LANDMARKS["RIGHT_HIP"]]) / 2
        
        # Z offset for feet
        foot_z = min(targets.get("L_foot", [0,0,0])[2], 
                     targets.get("R_foot", [0,0,0])[2])
        z_offset = -foot_z + 0.02
        
        # Create target transforms
        target_transforms = {}
        scaled_targets = {}
        print("\nTarget positions:")
        for name, pos in targets.items():
            scaled_pos = np.array([
                (pos[0] - hip_center[0]) * scale,
                (pos[1] - hip_center[1]) * scale,
                (pos[2] + z_offset) * scale
            ])
            target_transforms[name] = pin.SE3(np.eye(3), scaled_pos)
            scaled_targets[name] = scaled_pos
            print(f"  {name}: [{scaled_pos[0]:.3f}, {scaled_pos[1]:.3f}, {scaled_pos[2]:.3f}]")
        
        # Setup visualizer
        if visualize:
            viz_ok = self.visualizer.setup()
            if viz_ok:
                self.visualizer.add_target_markers(scaled_targets)
        
        # IK parameters
        q = self.q.copy()
        eps = 1e-2
        dt = 0.1
        damp = 1e-4
        
        print(f"\nRunning IK...")
        avg_error = float('inf')
        
        for iteration in range(max_iterations):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            total_error = 0.0
            dq = np.zeros(self.model.nv)
            
            for name, target_se3 in target_transforms.items():
                frame_name = valid_frames[name]
                frame_id = self.get_frame_id(frame_name)
                
                current_se3 = self.data.oMf[frame_id]
                pos_error = target_se3.translation - current_se3.translation
                total_error += np.linalg.norm(pos_error)
                
                J = pin.computeFrameJacobian(
                    self.model, self.data, q, frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )[:3, :]
                
                JJt = J @ J.T + damp * np.eye(3)
                dq += J.T @ np.linalg.solve(JJt, pos_error)
            
            dq_norm = np.linalg.norm(dq)
            if dq_norm > 1.0:
                dq = dq / dq_norm
            
            avg_error = total_error / len(target_transforms)
            
            if avg_error < eps:
                print(f"✓ Converged at iteration {iteration}, error: {avg_error:.4f}m")
                break
            
            q = pin.integrate(self.model, q, dt * dq)
            
            if visualize and self.visualizer.viz and iteration % 10 == 0:
                self.visualizer.display(q)
        
        if avg_error >= eps:
            print(f"⚠ Did not converge. Final error: {avg_error:.4f}m")
        
        if visualize and self.visualizer.viz:
            self.visualizer.display(q)
        
        self.q = q
        return q
    
    def _estimate_height(self, landmarks_3d: np.ndarray) -> float:
        """Estimate human height from landmarks"""
        head = landmarks_3d[MP_LANDMARKS["NOSE"]]
        ankle_center = (landmarks_3d[MP_LANDMARKS["LEFT_ANKLE"]] + 
                        landmarks_3d[MP_LANDMARKS["RIGHT_ANKLE"]]) / 2
        height = abs(head[2] - ankle_center[2]) + 0.15
        return height if height > 0.5 else 1.7
    
    def get_joint_positions(self) -> dict:
        """Get joint positions as dictionary"""
        result = {}
        if self.model.nq >= 7:
            result["base_position"] = self.q[:3].tolist()
            result["base_quaternion"] = self.q[3:7].tolist()
        
        idx = 7 if self.model.nq >= 7 else 0
        for i in range(1, self.model.njoints):
            name = self.model.names[i]
            joint = self.model.joints[i]
            nq = joint.nq
            if nq > 0 and idx < len(self.q):
                result[name] = float(self.q[idx]) if nq == 1 else self.q[idx:idx+nq].tolist()
                idx += nq
        return result


def visualize_detection(frame: np.ndarray, result):
    """Visualize MediaPipe pose on image"""
    from mediapipe_pose import SkeletonRenderer
    
    display = frame.copy()
    
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        SkeletonRenderer.draw_skeleton(display, landmarks)
        SkeletonRenderer.draw_landmarks(display, landmarks)
    
    max_dim = 800
    h, w = display.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(display, (int(w*scale), int(h*scale)))
    
    cv2.imshow("MediaPipe Pose Detection", display)
    print("\nPress any key to continue to G1 retargeting...")
    cv2.waitKey(0)


def process_image(image_path: str):
    """Process image for pose estimation and retargeting"""
    from mediapipe_pose import PoseDetector, PoseDataCollector
    
    print(f"\n{'='*60}")
    print(f"IMAGE TO G1 RETARGETING")
    print(f"{'='*60}")
    print(f"Image: {image_path}\n")
    
    # 1. Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: MediaPipe model not found: {MODEL_PATH}")
        return None
    
    if not os.path.exists(URDF_PATH):
        print(f"ERROR: G1 URDF not found: {URDF_PATH}")
        g1_dir = "unitree_ros/robots/g1_description"
        if os.path.exists(g1_dir):
            print("\nAvailable G1 URDFs:")
            for f in sorted(os.listdir(g1_dir)):
                if f.endswith('.urdf'):
                    print(f"  - {g1_dir}/{f}")
        return None
    
    # 2. Setup MediaPipe
    detector = PoseDetector(model_path=MODEL_PATH)
    if not detector.setup(running_mode='IMAGE'):
        print("Failed to setup pose detector")
        return None
    
    # 3. Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image")
        return None
    
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # 4. Detect pose
    result = detector.detect(frame)
    detector.close()
    
    if not result or not result.pose_world_landmarks:
        print("No pose detected!")
        cv2.imshow("No Pose", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
    
    print("✓ Pose detected!")
    
    # 5. Extract landmarks
    world_landmarks = result.pose_world_landmarks[0]
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    landmarks_robotics = PoseDataCollector.mediapipe_to_robotics(landmarks_np)
    
    print(f"\nKey landmarks (meters):")
    for name, idx in [("L_Wrist", 15), ("R_Wrist", 16), ("L_Ankle", 27), ("R_Ankle", 28)]:
        p = landmarks_robotics[idx]
        print(f"  {name}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
    
    # 6. Show MediaPipe detection
    print("\n--- STEP 1: MediaPipe Detection ---")
    visualize_detection(frame, result)
    
    # 7. Retarget to G1
    print("\n--- STEP 2: G1 Retargeting ---")
    try:
        retargeter = G1Retargeter(URDF_PATH)
        robot_q = retargeter.retarget(landmarks_robotics, visualize=True)
        
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        
        joint_positions = retargeter.get_joint_positions()
        print(f"\nJoint angles (radians → degrees):")
        for name, value in joint_positions.items():
            if name.startswith("base_"):
                continue
            if isinstance(value, float):
                print(f"  {name:30s}: {value:7.3f} rad ({np.degrees(value):7.1f}°)")
        
        print("\n✓ G1 robot visualization open in browser at http://127.0.0.1:7000/static/")
        print("  Press any key in OpenCV window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return {"landmarks": landmarks_robotics, "robot_q": robot_q, "joints": joint_positions}
        
    except Exception as e:
        print(f"Retargeting failed: {e}")
        import traceback
        traceback.print_exc()
        return {"landmarks": landmarks_robotics, "robot_q": None}


if __name__ == "__main__":
    # Check pinocchio first
    print("Checking Pinocchio installation...")
    pin, method = check_pinocchio()
    if pin is None:
        print("\nERROR: Pinocchio not installed!")
        print("Install with: pip install pin")
        sys.exit(1)
    
    print(f"✓ Pinocchio found (version: {getattr(pin, '__version__', 'unknown')})")
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "test_person.jpg"
        if not os.path.exists(img_path):
            print("\nUsage: python image_to_g1.py <image_path>")
            print("Example: python image_to_g1.py photo.jpg")
            sys.exit(1)
    
    result = process_image(img_path)
    
    if result and result.get("robot_q") is not None:
        print("\n" + "="*60)
        print("✓ SUCCESS! Both visualizations complete.")
        print("="*60)
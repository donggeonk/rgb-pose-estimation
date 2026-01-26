import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_hand_trajectories(pose_filename, timestamps_filename):
    """
    Plot right and left end effectors (wrist and ankle) trajectories over time
    
    Args:
        pose_filename: Path to pose .npy file
        timestamps_filename: Path to timestamps .npy file
    """
    # Load data
    pose_data = np.load(pose_filename)
    timestamps = np.load(timestamps_filename)
    
    print(f"Loaded: {pose_filename}")
    print(f"Shape: {pose_data.shape} (frames, landmarks, coordinates)")
    print(f"Total frames: {len(pose_data)}")
    
    # left_keypoint = pose_data[:, 15, :]   # left wrist
    # right_keypoint = pose_data[:, 16, :]  # right wrist

    left_keypoint = pose_data[:, 27, :]   # left ankle
    right_keypoint = pose_data[:, 28, :]  # right ankle
    
    # Convert timestamps to seconds (relative to start)
    time_seconds = (timestamps - timestamps[0]) / 1000.0
    
    # Calculate some statistics
    print(f"\n=== LEFT End Effector STATISTICS ===")
    print(f"X range: [{np.min(left_keypoint[:, 0]):.3f}, {np.max(left_keypoint[:, 0]):.3f}] meters")
    print(f"Y range: [{np.min(left_keypoint[:, 1]):.3f}, {np.max(left_keypoint[:, 1]):.3f}] meters")
    print(f"Z range: [{np.min(left_keypoint[:, 2]):.3f}, {np.max(left_keypoint[:, 2]):.3f}] meters")
    
    print(f"\n=== RIGHT End Effector STATISTICS ===")
    print(f"X range: [{np.min(right_keypoint[:, 0]):.3f}, {np.max(right_keypoint[:, 0]):.3f}] meters")
    print(f"Y range: [{np.min(right_keypoint[:, 1]):.3f}, {np.max(right_keypoint[:, 1]):.3f}] meters")
    print(f"Z range: [{np.min(right_keypoint[:, 2]):.3f}, {np.max(right_keypoint[:, 2]):.3f}] meters")
    
    # Create figure with 6 subplots (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    fig.suptitle('Trajectories Over Time', fontsize=14, fontweight='bold')
    
    # Left column: Right End Effector
    # Right End Effector X
    axes[0, 0].plot(time_seconds, right_keypoint[:, 0], 'b-', linewidth=1.5)
    axes[0, 0].set_title('Right End Effector - X Coordinate (Left/Right)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('X Position (meters)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Origin')
    axes[0, 0].legend()
    
    # Right End Effector Y
    axes[1, 0].plot(time_seconds, right_keypoint[:, 1], 'g-', linewidth=1.5)
    axes[1, 0].set_title('Right End Effector - Y Coordinate (Up/Down)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Y Position (meters)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Hip Level')
    axes[1, 0].legend()
    
    # Right End Effector Z
    axes[2, 0].plot(time_seconds, right_keypoint[:, 2], 'r-', linewidth=1.5)
    axes[2, 0].set_title('Right End Effector - Z Coordinate (Depth)', fontweight='bold')
    axes[2, 0].set_xlabel('Time (seconds)')
    axes[2, 0].set_ylabel('Z Position (meters)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Hip Depth')
    axes[2, 0].legend()
    
    # Right column: Left End Effector
    # Left End Effector X
    axes[0, 1].plot(time_seconds, left_keypoint[:, 0], 'b-', linewidth=1.5)
    axes[0, 1].set_title('Left End Effector - X Coordinate (Left/Right)', fontweight='bold')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('X Position (meters)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Origin')
    axes[0, 1].legend()
    
    # Left End Effector Y
    axes[1, 1].plot(time_seconds, left_keypoint[:, 1], 'g-', linewidth=1.5)
    axes[1, 1].set_title('Left End Effector - Y Coordinate (Up/Down)', fontweight='bold')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Y Position (meters)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Hip Level')
    axes[1, 1].legend()
    
    # Left End Effector Z
    axes[2, 1].plot(time_seconds, left_keypoint[:, 2], 'r-', linewidth=1.5)
    axes[2, 1].set_title('Left End Effector - Z Coordinate (Depth)', fontweight='bold')
    axes[2, 1].set_xlabel('Time (seconds)')
    axes[2, 1].set_ylabel('Z Position (meters)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Hip Depth')
    axes[2, 1].legend()
    
    # Calculate smoothness metrics (velocity changes)
    print(f"\n=== SMOOTHNESS ANALYSIS ===")
    
    # Calculate velocities (first derivative)
    dt = np.diff(time_seconds)
    
    right_keypoint_vel = np.diff(right_keypoint, axis=0) / dt[:, np.newaxis]
    left_keypoint_vel = np.diff(left_keypoint, axis=0) / dt[:, np.newaxis]
    
    # Calculate accelerations (second derivative - indicator of jerkiness)
    right_keypoint_acc = np.diff(right_keypoint_vel, axis=0) / dt[:-1, np.newaxis]
    left_keypoint_acc = np.diff(left_keypoint_vel, axis=0) / dt[:-1, np.newaxis]
    
    print(f"Right Wrist:")
    print(f"  Avg velocity magnitude: {np.mean(np.linalg.norm(right_keypoint_vel, axis=1)):.3f} m/s")
    print(f"  Max velocity magnitude: {np.max(np.linalg.norm(right_keypoint_vel, axis=1)):.3f} m/s")
    print(f"  Avg acceleration magnitude: {np.mean(np.linalg.norm(right_keypoint_acc, axis=1)):.3f} m/s²")
    
    print(f"\nLeft Wrist:")
    print(f"  Avg velocity magnitude: {np.mean(np.linalg.norm(left_keypoint_vel, axis=1)):.3f} m/s")
    print(f"  Max velocity magnitude: {np.max(np.linalg.norm(left_keypoint_vel, axis=1)):.3f} m/s")
    print(f"  Avg acceleration magnitude: {np.mean(np.linalg.norm(left_keypoint_acc, axis=1)):.3f} m/s²")

    plt.tight_layout()
    plt.show()


def main():
    output_dir = 'output'
    pose_files = glob.glob(os.path.join(output_dir, 'realtime_pose_3d_coordinate_20251021_202404.npy'))
    
    if not pose_files:
        print(f"No pose files found in {output_dir}/")
        return
    
    latest_pose_file = max(pose_files, key=os.path.getctime)
    
    # Get corresponding timestamp file
    base_name = os.path.basename(latest_pose_file)
    timestamp_str = base_name.replace('realtime_pose_3d_coordinate_', '').replace('.npy', '')
    latest_timestamps_file = os.path.join(output_dir, f'realtime_pose_timestamps_{timestamp_str}.npy')
    
    if not os.path.exists(latest_timestamps_file):
        print(f"Error: Timestamps file not found: {latest_timestamps_file}")
        return
    
    print("="*60)
    print("HAND TRAJECTORY ANALYSIS")
    print("="*60)
    print(f"Pose file: {latest_pose_file}")
    print(f"Timestamps file: {latest_timestamps_file}")
    print("="*60)
    
    plot_hand_trajectories(latest_pose_file, latest_timestamps_file)


if __name__ == "__main__":
    main()
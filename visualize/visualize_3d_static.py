import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob


def visualize_3d_pose_static(pose_filename, frame_num=0):
    """
    Visualize a single frame of 3D pose data with corrected coordinate system
    
    Args:
        pose_filename: Path to .npy file
        frame_num: Which frame to visualize (default: 0)
    """
    # Load data
    pose_data = np.load(pose_filename)
    
    print(f"Loaded: {pose_filename}")
    print(f"Shape: {pose_data.shape} (frames, landmarks, coordinates)")
    print(f"Visualizing frame {frame_num}/{pose_data.shape[0]-1}")
    
    # Get single frame - NO SWAPPING, use original MediaPipe coordinates
    # X = right/left (width)
    # Y = up/down (height)
    # Z = forward/backward (depth)
    frame = pose_data[frame_num]  # Shape: (33, 3)
    
    # Skeleton connections
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
        (11, 23), (12, 24), (23, 24),  # Torso
        (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face left
        (0, 4), (4, 5), (5, 6), (6, 8),  # Face right
        (27, 29), (27, 31), (28, 30), (28, 32),  # Feet
        (15, 17), (15, 19), (15, 21), (17, 19),  # Left hand
        (16, 18), (16, 20), (16, 22), (18, 20)   # Right hand
    ]
    
    # Landmark names for key points
    landmark_names = {
        0: "NOSE",
        11: "L_SHOULDER", 12: "R_SHOULDER",
        13: "L_ELBOW", 14: "R_ELBOW",
        15: "L_WRIST", 16: "R_WRIST",
        23: "L_HIP", 24: "R_HIP",
        25: "L_KNEE", 26: "R_KNEE",
        27: "L_ANKLE", 28: "R_ANKLE"
    }
    
    # Create figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates (original MediaPipe)
    xs = frame[:, 0]  # X = width (right/left)
    ys = frame[:, 1]  # Y = height (up/down)
    zs = frame[:, 2]  # Z = depth (forward/backward)
    
    # Plot landmarks
    ax.scatter(xs, ys, zs, c='red', marker='o', s=100, label='Landmarks')
    
    # Plot skeleton connections
    for start_idx, end_idx in connections:
        ax.plot(
            [xs[start_idx], xs[end_idx]],
            [ys[start_idx], ys[end_idx]],
            [zs[start_idx], zs[end_idx]],
            'b-', linewidth=2
        )
    
    # Add labels for key landmarks
    for idx, name in landmark_names.items():
        ax.text(xs[idx], ys[idx], zs[idx], f'  {name}', fontsize=8)
    
    # Set labels (original MediaPipe convention)
    ax.set_xlabel('X (meters) - Right/Left', fontsize=12)
    ax.set_ylabel('Y (meters) - Up/Down', fontsize=12)
    ax.set_zlabel('Z (meters) - Depth (Forward)', fontsize=12)
    ax.set_title(f'3D Pose Visualization - Frame {frame_num}', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle - rotated so person stands upright
    # Y-axis vertical, X and Z horizontal
    ax.view_init(elev=-75, azim=-90)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to visualize static pose data
    """
    # Find most recent pose file
    output_dir = 'output'
    pose_files = glob.glob(os.path.join(output_dir, 'realtime_pose_3d_coordinate_20251011_215041.npy'))
    
    if not pose_files:
        print(f"No pose files found in {output_dir}/")
        return
    
    # Get most recent file
    latest_file = max(pose_files, key=os.path.getctime)
    
    # Load to check number of frames
    pose_data = np.load(latest_file)
    num_frames = pose_data.shape[0]
    
    print("="*60)
    print("3D POSE STATIC VISUALIZATION")
    print("="*60)
    print(f"Found {len(pose_files)} pose file(s)")
    print(f"Using: {latest_file}")
    print(f"Total frames available: 0 to {num_frames - 1}")
    print("="*60)
    
    # Ask for frame number
    frame_input = input(f"Enter frame number (0-{num_frames-1}, default 0): ").strip()
    frame_num = int(frame_input) if frame_input else 0
    
    # Validate frame number
    if frame_num < 0 or frame_num >= num_frames:
        print(f"Invalid frame number! Using frame 0 instead.")
        frame_num = 0
    
    print(f"\nVisualizing frame {frame_num}...")
    print("You can rotate the 3D view with your mouse.")
    print("Close the window to exit.\n")
    
    visualize_3d_pose_static(latest_file, frame_num)


if __name__ == "__main__":
    main()
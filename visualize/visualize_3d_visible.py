# Add at the top with other imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import glob


def visualize_3d_pose_animation(pose_filename, fps=30, visibility_threshold=0.5):
    """
    Animate all frames of 3D pose data with corrected coordinate system
    
    Args:
        pose_filename: Path to .npy file
        fps: Frames per second for animation
        visibility_threshold: Minimum visibility score to display landmark (0-1)
    """
    # Load data
    pose_data = np.load(pose_filename)
    
    # Try to load visibility data if it exists
    visibility_file = pose_filename.replace('_3d_coordinate_', '_visibility_')
    visibility_data = None
    if os.path.exists(visibility_file):
        visibility_data = np.load(visibility_file)
        print(f"Loaded visibility data: {visibility_file}")
    else:
        print("No visibility data found - showing all landmarks")
    
    print(f"Loaded: {pose_filename}")
    print(f"Shape: {pose_data.shape} (frames, landmarks, coordinates)")
    print(f"Creating animation with {pose_data.shape[0]} frames at {fps} FPS")
    
    # NO SWAPPING - use original MediaPipe coordinates
    # X = right/left (width)
    # Y = up/down (height) 
    # Z = depth (forward/backward)
    
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
    
    # Create figure with smaller size (same as static)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate global limits for consistent view
    all_xs = pose_data[:, :, 0].flatten()
    all_ys = pose_data[:, :, 1].flatten()
    all_zs = pose_data[:, :, 2].flatten()
    
    max_range = np.array([all_xs.max()-all_xs.min(), 
                          all_ys.max()-all_ys.min(), 
                          all_zs.max()-all_zs.min()]).max() / 2.0
    mid_x = (all_xs.max()+all_xs.min()) * 0.5
    mid_y = (all_ys.max()+all_ys.min()) * 0.5
    mid_z = (all_zs.max()+all_zs.min()) * 0.5
    
    def update(frame_num):
        ax.clear()
        
        # Get frame data
        frame = pose_data[frame_num]
        xs = frame[:, 0]  # X = width (right/left)
        ys = frame[:, 1]  # Y = height (up/down)
        zs = frame[:, 2]  # Z = depth (forward/backward)
        
        # Get visibility for this frame if available
        if visibility_data is not None:
            visibility = visibility_data[frame_num]
            visible_mask = visibility >= visibility_threshold
        else:
            visible_mask = np.ones(len(xs), dtype=bool)  # Show all if no visibility data
        
        # Plot only visible landmarks
        visible_xs = xs[visible_mask]
        visible_ys = ys[visible_mask]
        visible_zs = zs[visible_mask]
        ax.scatter(visible_xs, visible_ys, visible_zs, c='red', marker='o', s=100)
        
        # Plot occluded landmarks in gray (optional)
        if visibility_data is not None:
            occluded_mask = ~visible_mask
            occluded_xs = xs[occluded_mask]
            occluded_ys = ys[occluded_mask]
            occluded_zs = zs[occluded_mask]
            ax.scatter(occluded_xs, occluded_ys, occluded_zs, c='gray', marker='x', s=50, alpha=0.3)
        
        # Plot skeleton connections (only if both endpoints are visible)
        for start_idx, end_idx in connections:
            if visible_mask[start_idx] and visible_mask[end_idx]:
                ax.plot(
                    [xs[start_idx], xs[end_idx]],
                    [ys[start_idx], ys[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    'b-', linewidth=2
                )
            elif visibility_data is not None:
                # Draw occluded connections as dashed gray lines
                ax.plot(
                    [xs[start_idx], xs[end_idx]],
                    [ys[start_idx], ys[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    'gray', linewidth=1, linestyle='--', alpha=0.3
                )
        
        # Set consistent limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Labels (original MediaPipe convention)
        ax.set_xlabel('X (meters) - Right/Left', fontsize=10)
        ax.set_ylabel('Y (meters) - Up/Down', fontsize=10)
        ax.set_zlabel('Z (meters) - Depth', fontsize=10)
        
        # Add visibility info to title
        if visibility_data is not None:
            visible_count = np.sum(visible_mask)
            ax.set_title(f'3D Pose Animation - Frame {frame_num}/{pose_data.shape[0]-1} ({visible_count}/33 visible)', 
                         fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'3D Pose Animation - Frame {frame_num}/{pose_data.shape[0]-1}', 
                         fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # Set viewing angle - tilted for better perspective (same as static)
        ax.view_init(elev=-75, azim=-90)
        
        return ax,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=pose_data.shape[0], 
                        interval=1000/fps, blit=False)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to visualize pose data
    """
    # Find most recent pose file
    output_dir = 'output'
    pose_files = glob.glob(os.path.join(output_dir, 'realtime_pose_3d_coordinate_*.npy'))
    
    if not pose_files:
        print(f"No pose files found in {output_dir}/")
        return
    
    # Get most recent file
    latest_file = max(pose_files, key=os.path.getctime)
    
    print("="*60)
    print("3D POSE ANIMATION")
    print("="*60)
    print(f"Found {len(pose_files)} pose file(s)")
    print(f"Using: {latest_file}")
    print("="*60)
    
    # Ask for FPS
    fps_input = input("Enter FPS for animation (default 30, press Enter to skip): ").strip()
    fps = int(fps_input) if fps_input else 30
    
    # Ask for visibility threshold
    visibility_input = input("Enter visibility threshold 0-1 (default 0.5, press Enter to skip): ").strip()
    visibility_threshold = float(visibility_input) if visibility_input else 0.5
    
    print(f"\nStarting animation at {fps} FPS...")
    print(f"Visibility threshold: {visibility_threshold}")
    print("Close the window to exit.\n")
    
    visualize_3d_pose_animation(latest_file, fps, visibility_threshold)


if __name__ == "__main__":
    main()
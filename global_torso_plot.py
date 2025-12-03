"""
Simple plot of X, Y, Z global coordinates over time.
Usage: python global_pose_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os


SESSION_DIR = "output/global_20251203_020813"

def plot_xyz(session_dir: str):
    if not os.path.exists(session_dir):
        print(f"Error: Directory not found: {session_dir}")
        return
    
    positions_path = os.path.join(session_dir, 'global_position.npy')
    if not os.path.exists(positions_path):
        print(f"Error: global_position.npy not found in {session_dir}")
        return
    
    positions = np.load(positions_path)
    
    # Time axis (assume 30 FPS if no timestamps)
    timestamps_path = os.path.join(session_dir, 'timestamps.npy')
    if os.path.exists(timestamps_path):
        timestamps = np.load(timestamps_path)
        time_sec = (timestamps - timestamps[0]) / 1000.0
    else:
        time_sec = np.arange(len(positions)) / 30.0
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    labels = ['X (Lateral)', 'Y (Vertical)', 'Z (Forward)']
    colors = ['red', 'green', 'blue']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time_sec, positions[:, i], color=color, linewidth=1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_ylabel(f'{label}\n(meters)')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    axes[0].set_title(f'Global Position - {os.path.basename(session_dir)}')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(session_dir, 'xyz_plot.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_xyz(SESSION_DIR)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from st_gcn_refiner import PoseRefinerModel

class PoseRefinementDataset(Dataset):
    def __init__(self, mediapipe_dir, mocap_dir, sequence_length=60):
        """
        mediapipe_dir: Folder containing .npy files from MediaPipe
        mocap_dir: Folder containing .npy files from MoCap (Ground Truth)
        sequence_length: Number of frames per sample (temporal window)
        """
        self.seq_len = sequence_length
        self.samples = []
        
        # 1. Load and pair files
        # Assuming file names match: "dance1.npy" in both folders
        mp_files = sorted(os.listdir(mediapipe_dir))
        
        for f_name in mp_files:
            if not f_name.endswith('.npy'): continue
            
            mp_path = os.path.join(mediapipe_dir, f_name)
            gt_path = os.path.join(mocap_dir, f_name)
            
            if not os.path.exists(gt_path):
                print(f"Warning: Missing GT for {f_name}")
                continue
                
            # Load data
            # Shape: (Frames, Joints, 3)
            mp_data = np.load(mp_path) 
            gt_data = np.load(gt_path)
            
            # 2. Basic Validation & Trimming
            min_len = min(len(mp_data), len(gt_data))
            mp_data = mp_data[:min_len]
            gt_data = gt_data[:min_len]
            
            # 3. Create sliding windows
            # We slice the long video into smaller chunks for training
            num_windows = min_len // sequence_length
            for i in range(num_windows):
                start = i * sequence_length
                end = start + sequence_length
                self.samples.append({
                    'input': mp_data[start:end],
                    'target': gt_data[start:end]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Convert to Float Tensor
        input_tensor = torch.from_numpy(sample['input']).float()
        target_tensor = torch.from_numpy(sample['target']).float()
        return input_tensor, target_tensor

def train():
    # --- Configuration ---
    MEDIAPIPE_DIR = "../mediapipe_output_processed" # Folder with just .npy arrays
    MOCAP_DIR = "../mocap_groundtruth"
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Data Setup ---
    dataset = PoseRefinementDataset(MEDIAPIPE_DIR, MOCAP_DIR, sequence_length=60)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training on {len(dataset)} sequences.")
    
    # --- Model Setup ---
    model = PoseRefinerModel(num_joints=33, in_channels=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loss Function: MPJPE (Mean Per Joint Position Error) + Smoothness
    l1_loss = nn.L1Loss()
    
    # --- Training Loop ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (noisy_pose, clean_pose) in enumerate(dataloader):
            noisy_pose = noisy_pose.to(DEVICE) # (B, T, V, 3)
            clean_pose = clean_pose.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            refined_pose = model(noisy_pose)
            
            # Calculate Loss
            # 1. Position Loss
            pos_loss = l1_loss(refined_pose, clean_pose)
            
            # 2. Velocity Loss (Temporal Smoothness) - helps with jitter
            # Calculate velocity (diff between frames)
            refined_vel = refined_pose[:, 1:] - refined_pose[:, :-1]
            clean_vel = clean_pose[:, 1:] - clean_pose[:, :-1]
            vel_loss = l1_loss(refined_vel, clean_vel)
            
            loss = pos_loss + (0.5 * vel_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")
        
    # Save Model
    torch.save(model.state_dict(), "st_gcn_refiner.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    train()
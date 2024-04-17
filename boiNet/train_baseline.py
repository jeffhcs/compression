import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from model import TwoResAutoEncoder, BGAutoencoder
from data import FaceDataset
import sys


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = TwoResAutoEncoder(900, 100)
    model = BGAutoencoder(1000)

    model.to(torch.device(device))
    model.train()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize dataset and dataloader
    full_dataset = FaceDataset('norm_bounding_boxes.csv', '../celeba/img_align_celeba')

    # criterion = nn.MSELoss(reduction='none')
    criterion = nn.MSELoss()
    
    num_epochs = 20
    batch_size = 25
    # dataset_size = len(full_dataset)
    dataset_size = 100
    model_save_name = "boiNet.pt"


    # Create a subset of the full dataset (100 samples)
    # indices = np.random.permutation(len(full_dataset))[:100]
    indices = list(range(dataset_size))
    dataset = Subset(full_dataset, indices) 

    # dataset = FaceDataset('norm_bounding_boxes.csv', '../celeba/img_align_celeba')
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        # total_fg_loss = 0.0
        # total_bg_loss = 0.0
        total_loss = 0.0
        for images, faces, bboxs in dataloader:

            # Forward pass
            images = images.to(device)
            # faces = faces.to(device)

            # fg_masks = create_fg_masks(bboxs)
            # fg_masks = fg_masks.to(device)

            optimizer.zero_grad()
            # fg_output, bg_output = model(images * (~fg_masks), faces)
            output = model(images)
            
            loss = criterion(images, output)
            
            # print(images.shape)
            # Calculate the loss for each region
            # fg_loss = fg_criterion(fg_output, faces)
            # bg_loss = bg_criterion(bg_output, images) * (~fg_masks)
            # bg_loss = fg_criterion(bg_output, images) 

            # Only consider masked areas by averaging non-zero entries
            # bg_loss = bg_loss.sum() / (~fg_masks).sum()

            # print(fg_loss, bg_loss)

            # Combine losses and perform backpropagation
            # total_loss = fg_loss + bg_loss
            loss.backward()
            optimizer.step()

            # Aggregate losses for logging
            total_loss += loss.item()
            # total_bg_loss += bg_loss.item()

        # Print epoch loss
        # avg_fg_loss = total_fg_loss / len(dataloader)
        # avg_bg_loss = total_bg_loss / len(dataloader)
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        sys.stdout.flush()

    
    torch.save(model, model_save_name)    

if __name__ == "__main__":
    # print("cuda" if torch.cuda.is_available() else "cpu")

    train()

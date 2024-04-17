import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, df_path, image_dir):
        self.dataframe = pd.read_csv(df_path)
        self.image_dir = image_dir

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((92, 84)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        bbox = self.dataframe.iloc[idx, 1:5].values
        face_crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        
        if self.transform:
            face_crop = self.transform(face_crop)

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        
        bbox = np.array(bbox, dtype=int)
        
        return image, face_crop, torch.tensor(bbox)


if __name__ == "__main__":
    # Load the bounding box data
    df_path = 'norm_bounding_boxes.csv'

    # Initialize dataset and dataloader
    dataset = FaceDataset(df_path, '../celeba/img_align_celeba')
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False)
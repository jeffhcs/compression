import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, df_path, image_dir, return_im_num=False):
        self.dataframe = pd.read_csv(df_path)
        self.image_dir = image_dir

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((92, 84)),
            transforms.ToTensor()
        ])
        
        self.return_im_num = return_im_num
    
    def __len__(self):
        return len(self.dataframe)
    
    def extract_im_num(self, filename: str):
        format_i = filename.find('.jpg')
        
        if format_i == -1:
            raise ValueError("Unexpected dataset file format")
        
        return torch.tensor(int(filename[:format_i]), dtype=torch.int)
    
    def __getitem__(self, idx):        
        filename = self.dataframe.iloc[idx, 0]
        img_name = os.path.join(self.image_dir, filename)
        
        image = Image.open(img_name)
        bbox = self.dataframe.iloc[idx, 1:5].values
        face_crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        if self.transform:
            face_crop = self.transform(face_crop)

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        
        bbox = np.array(bbox, dtype=int)
        
        if self.return_im_num:
            return self.extract_im_num(filename), image, face_crop, torch.tensor(bbox)
        
        return image, face_crop, torch.tensor(bbox)
    
    
class FaceAndAttributeDataset():
    def __init__(self, df_path, image_dir, attribute):
        self.dataframe = pd.read_csv(df_path)
        
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.attribute = attribute
    
    def __len__(self):
        return len(self.dataframe)
    
    def extract_im_num(self, filename: str):
        format_i = filename.find('.jpg')
        
        if format_i == -1:
            raise ValueError("Unexpected dataset file format")
        
        return torch.tensor(int(filename[:format_i]), dtype=torch.int)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        attribute_val = self.dataframe.iloc[self.extract_im_num(img_name)][self.atribute]
                
        image = Image.open(img_name)
        
        return image, torch.tensor(attribute_val == 1, dtype=torch.bool)
    
    
class JpgBeforeAfterDataset():
    def __init__(self, before_dir, after_dir): 
        # Assume after_dir is a subset of before_dir.      
        self.image_names = os.listdir(after_dir)
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
                
        before = Image.open(os.path.join(self.before_dir, img_name))
        after = Image.open(os.path.join(self.after_dir, img_name))
        
        return self.to_tensor(before), self.to_tensor(after)


if __name__ == "__main__":
    # Load the bounding box data
    df_path = 'norm_bounding_boxes.csv'

    # Initialize dataset and dataloader
    dataset = FaceDataset(df_path, '../celeba/img_align_celeba')
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False)
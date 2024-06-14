import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np


class KittiDataset(Dataset):
    

    def __init__(self, lidar_dir, intensity_dir,label_dir,x_dir, y_dir, z_dir):

        self.lidar_dir = lidar_dir
        self.intensity_dir = intensity_dir
        self.label_dir = label_dir 
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.z_dir = z_dir

        self.lidar_images = os.listdir(lidar_dir)
        self.intensity_images = os.listdir(intensity_dir)
        self.label_images = os.listdir(label_dir)
        self.x_images = os.listdir(x_dir)
        self.y_images = os.listdir(y_dir)
        self.z_images = os.listdir(z_dir)


        

    def __len__(self):
        return len(self.lidar_images)

    def __getitem__(self, index):
        lidar_path = os.path.join(self.lidar_dir, self.lidar_images[index])
        intensity_path = os.path.join(self.intensity_dir, self.intensity_images[index])
        label_path = os.path.join(self.label_dir , self.label_images[index])
        x_path = os.path.join(self.x_dir , self.x_images[index])
        y_path = os.path.join(self.y_dir , self.y_images[index])
        z_path = os.path.join(self.z_dir , self.z_images[index])
        
        

        # Load lidar data with channel dimension added
        lidar = np.load(lidar_path)
        lidar = np.expand_dims(lidar, axis=0)  # Add channel dimension
        lidar = torch.from_numpy(lidar).float()

        # Load x data with channel dimension added
        x = np.load(x_path)
        x = np.expand_dims(x, axis=0)  # Add channel dimension
        x = torch.from_numpy(x).float()

        # Load y data with channel dimension added
        y = np.load(y_path)
        y = np.expand_dims(y, axis=0)  # Add channel dimension
        y = torch.from_numpy(y).float()

        # Load z data with channel dimension added
        z = np.load(z_path)
        z = np.expand_dims(z, axis=0)  # Add channel dimension
        z = torch.from_numpy(z).float()


        # Load intensity data with channel dimension added
        intensity = np.load(intensity_path)
        intensity = np.expand_dims(intensity, axis=0)  # Add channel dimension
        intensity = torch.from_numpy(intensity).float()

        # Load label data with channel dimension added
        label = np.load(label_path)
        #label = np.expand_dims(label, axis=0)  # Add channel dimension
        label = torch.from_numpy(label).long()


        
        concatenated_img = torch.cat((x,y,z,lidar), dim=0) #T1 #in_channels=4
        #concatenated_img = torch.cat((x,y,z,lidar, intensity), dim=0) #T2/3 #in_channels=5
      

        return concatenated_img,label

 
    
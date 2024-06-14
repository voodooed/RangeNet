#Predictibg label image from individual files
import torch
from PIL import Image
from model import Darknet
from model2 import UNET
import numpy as np
from torchvision.utils import save_image
from transform_utils import lidar_transform,intensity_transform


def save_as_rgb_image(output_tensor, save_path):
    # Assuming output_tensor is a 1xCxHxW tensor where C is 1
    
    # Convert to CPU and numpy, then scale to [0, 255]
    output_np = output_tensor.squeeze().cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)
    
    # Convert grayscale to RGB by replicating the single channel
    output_rgb = np.stack([output_np]*3, axis=-1)
    
    # Convert to PIL Image and save
    output_img = Image.fromarray(output_rgb, 'RGB')
    output_img.save(save_path)

# Prepare the input data
lidar_path = "/home/viveka21/projects/def-rkmishra-ab/viveka21/VoxelScape_Data/Test/test_lidar_depth/10_000000.jpg"
intensity_path = "/home/viveka21/projects/def-rkmishra-ab/viveka21/VoxelScape_Data/Test/test_lidar_intensity/10_000000.jpg"

# Load the input images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = Darknet(in_channels=1, out_channels=1)  # Initialize your model architecture
model = UNET(in_channels=1, out_channels=1)  # Initialize your model architecture
checkpoint = torch.load('/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 3/RangeNet/Output/T1/T1_model.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])

model.to(device)
model.eval()  # Switch the model to evaluation mode

lidar = Image.open(lidar_path).convert("L")
intensity = Image.open(intensity_path).convert("L")

       
intensity = intensity_transform(intensity) #For adding a channel
lidar = lidar_transform(lidar)
            
#concatenated_img = np.concatenate((lidar[..., np.newaxis], camera,incidence[..., np.newaxis]), axis=2)
#input_data = torch.cat((binary, label,incidence,lidar, color, rgb), dim=0)
input_data = lidar
input_data = input_data.unsqueeze(0) 

input_data = input_data.to(device)
with torch.no_grad():
    output = model(input_data)

# Check if the output tensor is on the GPU and move it to CPU
if output.is_cuda:
    output = output.cpu()

# Replicate the grayscale output across 3 channels to convert it to RGB
output_rgb = output.repeat(1, 3, 1, 1)  # Assuming output is in shape [1, 1, H, W]

# Save the image
save_path = '/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 3/RangeNet/Output/T1/output_image.jpg'
#save_image(output_rgb, save_path)
save_image(output, save_path)
#save_image(output, save_path)

# Example usage
#save_path = 'output_image.jpg'
#save_as_rgb_image(output, save_path)

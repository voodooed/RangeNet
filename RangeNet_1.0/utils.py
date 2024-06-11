import torch
import torchvision
from dataset import KittiDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train__lidar_dir,
    train_intensity_dir,
    train_label_dir,
    train_x_dir,  
    train_y_dir, 
    train_z_dir,  


    val_lidar_dir,
    val_intensity_dir,
    val_label_dir,
    val_x_dir,    
    val_y_dir,   
    val_z_dir,   

    test_lidar_dir,
    test_intensity_dir,
    test_label_dir,
    test_x_dir,  
    test_y_dir,   
    test_z_dir,  


    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_ds = KittiDataset(
        lidar_dir = train__lidar_dir,
        intensity_dir=train_intensity_dir,
        label_dir = train_label_dir,
        x_dir=train_x_dir, 
        y_dir=train_y_dir,  
        z_dir=train_z_dir,  

       
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = KittiDataset(
        lidar_dir = val_lidar_dir,
        intensity_dir=val_intensity_dir,
        label_dir = val_label_dir,
        x_dir=val_x_dir,  
        y_dir=val_y_dir,  
        z_dir=val_z_dir, 



    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = KittiDataset(
        lidar_dir=test_lidar_dir,
        intensity_dir=test_intensity_dir,
        label_dir = test_label_dir,
        x_dir=test_x_dir,  
        y_dir=test_y_dir,  
        z_dir=test_z_dir,  


    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, device="cuda"):
    model.eval()

    mse_loss = torch.nn.MSELoss(reduction='sum')
    total_mse = 0
    total_pixels = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = model(x)
            total_mse += mse_loss(preds, y)
            total_pixels += torch.numel(preds)

    print(
        f"Mean Squared Error: {total_mse/total_pixels:.4f}"
    )
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="/DATA2/Vivek/Code/Implementation/UNET_T2/saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            #preds = torch.sigmoid(model(x))
            preds = model(x)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.jpg"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.jpg")

    model.train()



def plot_losses(train_losses, val_losses, save_path="/DATA2/Vivek/Code/Implementation/UNET_T3/loss_plot.png"):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Train and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)  # Save the plot to a file
    plt.show()

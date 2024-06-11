import torch
import random
import config as config
import numpy as np
import pandas as pd
from model import Darknet
from train import train_fn,val_fn,test_fn
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    plot_losses
)
from dataset import KittiDataset 


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


def get_weights():

        weights = {
        0: 0.012435843358855257, 1: 0.07084600265353548, 2: 0.0036331568465433122,
        3: 0.0024746719284991833, 4: 0.03963860185398928, 5: 0.010680881905904226,
        6: 0.04319204406549489, 7: 0.008983132005921277, 8: 0.0028492674204722485,
        9: 0.24428430602039514, 10: 0.0032094812797592363, 11: 0.14095381885918182,
        12: 0.0, 13: 0.27130546831301566, 14: 0.008430416581490963, 15: 0.06369792522894488,
        16: 0.03320962649268567, 17: 0.006247318243276751, 18: 0.031939692185776705,
        19: 0.001986299257892859
    }
        weights = torch.Tensor(list(weights.values()))

        return weights

def main():
  
    model = Darknet(in_channels=config.in_channels , out_channels=config.out_channels).to(config.DEVICE)

    weights = get_weights().to(config.DEVICE)
    loss_fn = torch.nn.NLLLoss(weight=weights, ignore_index=-1)

    # Create the optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, val_loader, test_loader = get_loaders(
        config.TRAIN_Lidar_DIR, 
        config.TRAIN_Intensity_DIR,
        config.TRAIN_LABEL_DIR,
        config.TRAIN_X_DIR, 
        config.TRAIN_Y_DIR,
        config.TRAIN_Z_DIR, 

        config.VAL_Lidar_DIR, 
        config.VAL_Intensity_DIR,
        config.VAL_LABEL_DIR,
        config.VAL_X_DIR,
        config.VAL_Y_DIR, 
        config.VAL_Z_DIR, 

        config.TEST_Lidar_DIR, 
        config.TEST_Intensity_DIR,
        config.TEST_LABEL_DIR,
        config.TEST_X_DIR, 
        config.TEST_Y_DIR, 
        config.TEST_Z_DIR, 


        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("/DATA2/Vivek/Code/Implementation/UNET_T3/Phase_3/my_checkpoint.pth.tar"), model)


    #check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    # Create a DataFrame to hold your data
    df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Validation Loss'])
    best_val_loss = float('inf')  # start with a high loss

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1} of {config.NUM_EPOCHS}")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        val_loss = val_fn(val_loader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss: {train_losses[-1]}")
        print(f"Validation loss: {val_losses[-1]}")

        # Append the losses for this epoch to the DataFrame
        new_row = pd.DataFrame({'Epoch': [epoch+1], 'Train Loss': [train_loss], 'Validation Loss': [val_loss]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(config.Loss_CSV_Path, index=False) #Change Accordingly


        # save model if validation loss has decreased
        if val_loss < best_val_loss:
            print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(best_val_loss, val_loss))
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, config.Checkpoint_Path)
            best_val_loss = val_loss

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        #save_checkpoint(checkpoint,'my_checkpoint.pth.tar')

      
    # Test the model after training is done
    test_loss = test_fn(test_loader, model, loss_fn)
    print(f"Test loss: {test_loss}")

    # After all epochs are completed
    plot_losses(train_losses, val_losses, config.Loss_Plot_Path)  # Call the function to plot the losses
    



if __name__ == "__main__":
    main()

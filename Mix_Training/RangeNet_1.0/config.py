import torch
import os

#Things to Change
    #in_channels
    #Loss_Plot_Path
    #Loss_CSV_Path
    #Checkpoint_Path 
    #Concatenated image in dataset.py 

Trial_Num = "T1" #Change
Trial_Path = f"Objective_3/RangeNet++/MiX_Data_Training/Output_1.0/{Trial_Num}" #Change
if not os.path.exists(Trial_Path):
    os.makedirs(Trial_Path)

#Voxelscape Dataset

# Hyperparameters 
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
NUM_EPOCHS = 150
NUM_WORKERS = 4

out_channels = 20
in_channels = 5 #Change Accordingly


Loss_Plot_Path = f"{Trial_Path}/{Trial_Num}_loss_plot.png"
Loss_CSV_Path = f"{Trial_Path}/{Trial_Num}_loss.csv"
Checkpoint_Path = f"{Trial_Path}/{Trial_Num}_model.pth.tar"



PIN_MEMORY = True
LOAD_MODEL = False

base_path = "Kitti_Downstream_Task_Data_2500_Frames/"

TRAIN_Lidar_DIR = base_path + "Train/train_lidar_depth"
TRAIN_Intensity_DIR = base_path + "Train/train_lidar_intensity" 
#TRAIN_Intensity_DIR = base_path + "Train/train_lidar_cycle_intensity" #CycleGAN Intensity #T3
TRAIN_LABEL_DIR = base_path + "Train/train_lidar_label"
TRAIN_X_DIR = base_path + "Train/train_lidar_x"
TRAIN_Y_DIR = base_path + "Train/train_lidar_y"
TRAIN_Z_DIR = base_path + "Train/train_lidar_z"

VAL_Lidar_DIR = base_path + "Val/val_lidar_depth"
VAL_Intensity_DIR = base_path + "Val/val_lidar_intensity"
#VAL_Intensity_DIR = base_path + "Val/val_lidar_cycle_intensity" #CycleGAN Intensity
VAL_LABEL_DIR = base_path + "Val/val_lidar_label"
VAL_X_DIR = base_path + "Val/val_lidar_x"
VAL_Y_DIR = base_path + "Val/val_lidar_y"
VAL_Z_DIR = base_path + "Val/val_lidar_z"


TEST_Lidar_DIR = base_path + "Test/test_lidar_depth"
TEST_Intensity_DIR = base_path + "Test/test_lidar_intensity"
#TEST_Intensity_DIR = base_path + "Test/test_lidar_cycle_intensity" #CycleGAN Intensity
TEST_LABEL_DIR = base_path + "Test/test_lidar_label"
TEST_X_DIR = base_path + "Test/test_lidar_x"
TEST_Y_DIR = base_path + "Test/test_lidar_y"
TEST_Z_DIR = base_path + "Test/test_lidar_z"



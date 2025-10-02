#region libraries
import os
import cv2
from roboflow import Roboflow
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

#endregion
###################################################################################################

#region training a model

#download dataset from roboflow (contains train, val and test data)
def download_robo_dataset(path_to_datasets, robo_key, workspace, project, v_version):
    """
    Downloads a dataset from Roboflow.

    Parameters:
    path_to_datasets (str): Path where the dataset will be downloaded and saved.
    robo_key (str): Your Roboflow API key.
    workspace (str): The name of your Roboflow workspace.
    project (str): The name of your Roboflow project.
    v_version (int): The version number of your Roboflow project.
    """
    os.chdir(f'{path_to_datasets}') #make sure to save new model in spacified ROBO_dataset folder
    rf = Roboflow(api_key=robo_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(v_version).download('yolov8')


#fine-tune model locally
def train_YOLO(path_to_model, path_to_ft_model, path_to_robo_folder, epochs, b, patience):
    """
    Trains a YOLO model using the specified parameters.

    Parameters:
    path_to_model (str): Path to the base YOLO model to fine-tune (or another already fine-tuned model)
    path_to_ft_model (str): Directory where the fine-tuned model and results will be saved.
    path_to_robo_folder (str): Path to the Roboflow dataset folder containing data.yaml.
    epochs (int): Number of training epochs.
    b (int): Batch size for training.
    patience (int): Number of epochs with no improvement after which training will be stopped.
    """
    model = YOLO(f'{path_to_model}')
    robodataset=os.path.basename(os.path.normpath(path_to_robo_folder))
    newname=f'{robodataset}_batch{b}'

    os.chdir(f'{path_to_ft_model}')
    results = model.train(
        data=f'{path_to_robo_folder}\\data.yaml',
        imgsz=640,
        epochs=epochs,
        batch=b,
        patience=patience,
        name=newname)
            

#upload model back to roboflow (if more fine-tuning needed)
def upload_yolo_to_robo(robo_key, workspace, project, v_version, path_to_model_folder):
    """
    robo_key: str : your roboflow api key
    workspace: str : your roboflow workspace name
    project: str : your roboflow project name
    v_version: int : version number of your roboflow project
    path_to_model_folder: str : path to the folder where your model weights are stored
    """
    rf = Roboflow(api_key=robo_key)
    project = rf.workspace(workspace).project(project)
    version = project.version(v_version)
    version.deploy("yolov8", f'{path_to_model_folder}\\weights', 'best.pt')

#endregion
###################################################################################################

#region inference
def predict_yolo(model, path_to_img, conf, iou):
    """
    model: YOLO model object to use for prediction. model=YOLO('path_to_model')
    path_to_img: str, path to the image file to run inference on.
    conf: float, confidence threshold for predictions.
    iou: float, IoU threshold for non-max suppression.
    """
    #obtain model predictions
    results = model.predict(path_to_img, conf=conf, iou=iou, verbose=False)
    predictions = results[0].boxes.data.tolist() #first 4 are bounding box, 5th is confidence score and 6th is class
    return predictions

#endregion
###################################################################################################
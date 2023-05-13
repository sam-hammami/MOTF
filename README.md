# MOTF
# Object Detection using Detectron2

This project demonstrates how to perform object detection using the [Detectron2](https://github.com/facebookresearch/detectron2) library. 

## Introduction

Object detection is a computer vision technique for locating instances of objects in images or videos. Object detection algorithms typically leverage machine learning or deep learning to produce meaningful results.

## Setup and Installation
We used Colab for this project.
Before running the script, please ensure the following packages are installed:

- torch
- torchvision
- fvcore
- detectron2
- roboflow

You can install them using pip:

!pip install torch torchvision
!pip install git+https://github.com/facebookresearch/fvcore.git
!pip install -U 'git+https://github.com/facebookresearch/detectron2.git'
!pip install roboflow

Dataset
We use a dataset provided by Roboflow. To get the dataset, use the following curl command:
!curl -L "https://app.roboflow.com/ds/1dzd8Ppul6?key=1ux6V7eRyZ" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

The dataset is expected to be in COCO format.

Usage
To train the model, run the following command:
python object_detection.py

You can visualize the training process using TensorBoard. Load TensorBoard with:
%load_ext tensorboard
%tensorboard --logdir output

Results
The trained model is used to make predictions on the test dataset. 
The predictions are visualized using the Visualizer class from Detectron2.

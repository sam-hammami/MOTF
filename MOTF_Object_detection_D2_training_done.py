# First, we need to install necessary packages
# torch and torchvision are for PyTorch, which is a framework for machine learning
# fvcore is a lightweight core library by Facebook
# detectron2 is a library for object detection and segmentation
# roboflow is a tool to manage, preprocess, augment, and version datasets, and then to prepare them for training in machine learning frameworks

!pip install -U torch torchvision
!pip install git+https://github.com/facebookresearch/fvcore.git
!pip install -U 'git+https://github.com/facebookresearch/detectron2.git'
!pip install roboflow

# Importing necessary libraries and modules
import os
import json
import cv2
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()  # setting up logger for detectron2

# More imports from detectron2 library
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

# Importing the dataset from Roboflow, unzipping it, then deleting the zip file
!curl -L "https://app.roboflow.com/ds/1dzd8Ppul6?key=1ux6V7eRyZ" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# Registering the COCO instances for training, validation, and testing datasets
register_coco_instances("my_dataset_train", {}, "/content/train/_annotations.coco.json", "/content/train")
register_coco_instances("my_dataset_val", {}, "/content/valid/_annotations.coco.json", "/content/valid")
register_coco_instances("my_dataset_test", {}, "/content/test/_annotations.coco.json", "/content/test")

# Setting up detectron2 logger
setup_logger()

# Importing some common libraries and detectron2 utilities
import numpy as np
import random
from google.colab.patches import cv2_imshow
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

# Visualizing training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])

# Importing our own Trainer Module here to use the COCO validation evaluation during training
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Configuring the model for training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))  # Use a pre-configured model from the model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Use pre-trained weights
cfg.DATASETS.TRAIN = ("my_dataset_train",)  # Specify the training dataset
cfg.DATASETS.TEST = ("my_dataset_val",)  # Specify the validation dataset

# Setting up the data loader and solver parameters
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000

# Modify the solver steps based on your dataset size, let's train a bit longer
cfg.SOLVER.MAX_ITER = 3000 
cfg.SOLVER.STEPS = (2000, 2500)
cfg.SOLVER.GAMMA = 0.05

# Configure model testing parameters
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Increase if you have more GPU memory
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 93  # Set the number of classes based on your dataset

cfg.TEST.EVAL_PERIOD = 500

# Create the trainer, load the initial weights, and start training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Visualize training curves in TensorBoard
%load_ext tensorboard
%tensorboard --logdir output

# Configure the model for testing
cfg.MODEL.WEIGHTS = "/content/output/model_final.pth"  # Load the final weights after training
cfg.DATASETS.TEST = ("my_dataset_test", )  # Specify the testing dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_train")

# Configure model testing parameters for detection
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # Set a lower testing threshold for this model
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000  # Increase the number of proposals if you face missing detections

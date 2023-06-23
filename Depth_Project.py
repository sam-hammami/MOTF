import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import torch
from gtts import gTTS
from IPython.display import Audio
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

# Define the ranges for depth segmentation
RANGES = [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]

# Define verbal descriptions for each range
DISTANCE_DESCRIPTIONS = ["far", "mid-range", "close", "very close", "extremely close"]

# Define colors for each range
COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def segment_and_color_depth_map(depth_map, ranges, colors):
    color_map = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    counts = []

    for r, color in zip(ranges, colors):
        mask = (depth_map > r[0]) & (depth_map <= r[1])
        color_map[mask] = color
        counts.append(np.count_nonzero(mask))

    return color_map, counts

# Load the depth estimation model
MODEL_TYPE = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if MODEL_TYPE == "DPT_Large" or MODEL_TYPE == "DPT_Hybrid":
    transform_midas = midas_transforms.dpt_transform
else:
    transform_midas = midas_transforms.small_transform

# Load the object detection model
model_detection = fasterrcnn_resnet50_fpn(pretrained=True)
model_detection = model_detection.to(device)
model_detection.eval()

# Define the transformations for object detection
transform_detection = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),
    transforms.ToTensor()
])

def main():
    st.title("Depth Estimation and Object Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_rgb = np.array(image)

        # Display the original image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Measure the processing time
        start_time = time.time()

        # Apply transformations and run depth estimation
        input_batch = transform_midas(image_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        output = (output - output.min()) / (output.max() - output.min())

        # Apply transformations and run object detection
        input_batch_detection = transform_detection(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model_detection(input_batch_detection)

        pred_score = list(predictions[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > 0.5][-1]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions[0]['boxes'].detach().cpu().numpy())]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions[0]['labels'].cpu().numpy())]
        pred_class = pred_class[:pred_t + 1]

        color_map, counts = segment_and_color_depth_map(output, RANGES, COLORS)
        dominant_description = DISTANCE_DESCRIPTIONS[np.argmax(counts)]
        dominant_range_text = f"The dominant distance is: {dominant_description}"

        if pred_class:
            object_description = f"The object in front of you is a {pred_class[0]} and it's {dominant_description}"
            st.write(object_description)
            tts = gTTS(object_description)
            tts.save('object_description.wav')
            sound_file = 'object_description.wav'
            st.audio(sound_file, format='audio/wav')

        st.image(output, caption='Depth Map', use_column_width=True)
        st.image(color_map, caption='Color Map', use_column_width=True)

        end_time = time.time()
        st.write(f"Time taken to process image: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()

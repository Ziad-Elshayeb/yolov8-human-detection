# YOLOv8 Human Detection

This project demonstrates how to use a trained YOLOv8 model to detect humans in images or videos. The model has been trained using the **COCO (Common Objects in Context)** dataset, specifically on the human category, which is one of the 80 categories in the COCO dataset.

## Dataset: COCO

The **COCO dataset** was used as the source for the training data. The images were originally downloaded in **COCO format**, and the annotations were then **converted to YOLO format** and normalized. The YOLO format uses a simpler annotation structure that only includes the bounding box coordinates and the class ID.

### Conversion from COCO Format to YOLO Format:

In the COCO format, bounding box annotations are represented as:

[x_min, y_min, width, height]

Where:
- `x_min, y_min` are the coordinates of the top-left corner of the bounding box.
- `width, height` are the dimensions of the bounding box.

In the **YOLO format**, bounding box annotations are normalized and represented as:

[class_id, x_center, y_center, width, height]

Where:
- `x_center, y_center` are the coordinates of the center of the bounding box (normalized).
- `width, height` are the dimensions of the bounding box (normalized).

### Normalization Process:

The YOLO format normalizes the coordinates and dimensions by the image width and height. The conversion equations from COCO format to YOLO format are as follows:

- `x_center = (x_min + (width / 2)) / image_width`
- `y_center = (y_min + (height / 2)) / image_height`
- `width = width / image_width`
- `height = height / image_height`

This normalization ensures that the bounding box coordinates are between `0` and `1`, making the model adaptable to different image resolutions.

### Training Data

- **Dataset**: COCO 2017
- **Category used**: Human
- **Number of images**: The dataset includes thousands of images labeled with human instances.
- **Annotation Format**: The COCO dataset annotations were converted and normalized to YOLO format.

## Setup

To set up this project, first, install the required dependencies.

### Install Requirements

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/my-yolov8-human-detection.git
cd yolov8-human-detection

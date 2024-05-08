# 3D Object Detection in Single Image

The project aims to locate and recognize objects in three-dimensional space from a 2D image by leveraging deep learning techniques to infer depth and spatial relationships. 
It demonstrates real-time 3D object detection through the PyTorch implementation based on YOLOv5, showcasing its application in robotics perception within the ROS framework in the Gazebo simulator.

## Features
- **Yolo Detection:** Utilized the yolov5 algorithm to estimate 3D bounding boxes of objects in the KITTI dataset and integrated with Gazebo and ROS2 to track objects.
- **PyTorch Implementation:** Utilized PyTorch for training YOLOv5 on preprocessed KITTI dataset and developing ROS2 nodes for image capture, model inference, 3D bounding box predictions, object detection, tracking, and result publication.
- **Integration with ROS2 Framework:** Leveraged ROS2 for robot control, Gazebo integration, and node development, facilitating image capture, topic/message/service-based communication, and modular system adaptability.
- **Gazebo Simulator Usage:** Facilitates 3D object detection by providing realistic environment, incorporating relevant objects, and transmitting simulated sensor data, such as camera images, to ROS2 nodes for efficient processing.

## Dependencies
* Ubuntu 
* ROS Humble Hawksbill
* Gazebo 11 (gazebo_ros_pkgs)
* PyTorch (above 1.8)
* OpenCV
* Python (above 3.8)
* NumPy
* Anaconda

Tested on
* Ubuntu 22.04 LTS, ROS Humble Hawksbill (other dependencies in )

## Deployment Overview
The entire project has two sections-
1. Deep Learning Model Development (PyTorch, KITTI)
2. Robot Integration with Gazebo and ROS2

## 1. Deep Learning Model Development
- **Utilize Yolov5 Detection Model:** Modify the network layer and output layer to predict 3D bounding boxes (center coordinates, size, orientation) and confidence scores.
- **Train the Model on KITTI Dataset:**  Train the modified Yolov5 model on the preprocessed KITTI dataset for 3D object detection.
- **Evaluate Model Performance:** Monitor training progress with validation metrics and evaluate the final model on the testing set for generalizability assessment.

### How to run
To get started, follow these steps:

### Clone the Repository

```bash
# Clone Repository
git clone https://github.com/Hetal-patel1994/3d-single-object-detection.git
```

### 2. Navigate to project directory

```bash
cd heart-disease-diagnosis
```
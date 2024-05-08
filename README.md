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
* Ubuntu 22.04 LTS, ROS Humble Hawksbill, PyTorch 2.3.0, CUDA 11.8

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

### Set Up Environment

Create conda environment

```bash
conda create -n yolo3d python=3.10 numpy
```

Install PyTorch and torchvision version above 1.8 as per python and supported GPU

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Navigate to project directory

```bash
cd 3d-single-object-detection\src\yolo3d\scripts\
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Pretrained Weights

To run inference code or resuming training, you can download pretrained ResNet18 or VGG11 model. I have train model with 10 epoch each. You can download model with resnet18 or vgg11 for --weights arguments.

```bash
cd 3d-single-object-detection\src\yolo3d\scripts\weights
python get_weights.py --weights resnet18
```

### Inference

For inference with pretrained model you can run below code.

```bash
python inference.py \
    --weights yolov5s.pt \
    --source eval/image_2 \
    --reg_weights weights/resnet18.pkl \
    --model_select resnet18 \
    --output_path runs/ \
    --show_result --save_result
```

## 2. Robot Integration with Gazebo and ROS2
- **Develop a Simulated Environment in Gazebo:**  Design a Gazebo 3D environment with relevant detection objects (matching KITTI categories) and define a camera sensor model matching the robot's actual camera.
- **Develop ROS2 Nodes for Robot Integration:**  Develop ROS2 nodes for capturing camera images, performing model inference using YOLOv5 on these images, and publishing the resulting 3D bounding box predictions and confidence scores on a dedicated ROS topic.

### How to run
To get started, follow these steps:

### Navigate to project directory

```bash
cd 3d-single-object-detection
```

### Install Dependencies

```bash
colcon build
```

### Launch

Terminal-1: To load the Gazebo environment

```bash
source 3d-single-object-detection/install/setup.bash
ros2 launch gazebo_simulation simulation.launch.py
```

Terminal-2:  To load the detection inference

```bash
source 3d-single-object-detection/install/setup.bash
ros2 run yolo3d inference_ros.py
```

## Screenshots

<div>
<img src="screenshots/ss_3.PNG">
<img src="screenshots/ss_6.PNG">
<img src="screenshots/ss_8.PNG">

</div>
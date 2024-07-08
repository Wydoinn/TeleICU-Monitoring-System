# [TeleICU Monitoring System](https://github.com/Wydoinn/TeleICU-Monitoring-System)

![TeleICU Logo](https://img.shields.io/badge/TeleICU-Monitoring%20System-blue?style=for-the-badge&logo=github) <img src="https://github.com/Wydoinn/TeleICU-Monitoring-System/assets/120785316/129088f6-88c9-426e-817b-f977c4ed7043" alt="Intel Logo" width="auto" height="30">

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Overview](#overview)
- [Demo](#demo)
- [Key Resources](#key-resources)
- [Installation](#installation)
- [Testing](#testing)
- [Model Conversion](#model-conversion)
- [Model Classes](#model-classes)
- [Best Model Performance](#best-model-performance)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## 🔬 Problem Statement

**Innovative Monitoring System for TeleICU Patients Using Video Processing and Deep Learning**

TeleICU is concept for monitoring ICU patients from remote locations to reduce the burden of on-site intensivist. Currently there are multiple products available in this domain where one profession seating at remote location physically monitors one or two remote patients in TeleICU. The proposed solution should work to reduce the burden of remote health care professional so, one remote health care professional can monitor 5 or more patients at single time.

## 🔍 Overview

TeleICU is an innovative remote monitoring system that empowers intensivists to manage more patients efficiently. By leveraging cutting-edge AI technologies, it revolutionizes critical care:

- **YOLOv10**: A powerful deep learning model that identifies patients and tracks vitals in real-time video feeds.
- **Deep SORT**: Builds on YOLOv10's capabilities, tracking patient movements to create a comprehensive health picture.

### 🌟 Key Benefits

- Enables a single intensivist to monitor multiple patients simultaneously
- Facilitates faster interventions through real-time monitoring
- Potentially improves patient outcomes
- Boosts overall efficiency and effectiveness in critical care

## 🎥 Demo

[View Demo Video](https://github.com/Wydoinn/TeleICU-Monitoring-System/assets/120785316/82fc5ca4-63f4-4489-8aab-ebe7e5697443)

## 🔗 Key Resources

- [Dataset Download](https://drive.google.com/drive/folders/1HSTfpo4IAEo9k5aSaw5KK92__wk-zGVT?usp=sharing)
- [Evaluation Metrics](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/evaluation)
- [Training Results](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/examine)
- [Training Notebooks](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/trains)
- [Predicted Output](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/output)

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Wydoinn/TeleICU-Monitoring-System.git
cd TeleICU-Monitoring-System
```

### 2. Create a New Environment

#### Using Conda

```bash
conda env create -f conda.yml
conda activate teleicu-monitoring-system
pip install -r requirements.txt
```

#### Using Pip

```bash
python -m virtualenv -p python3.11.7 teleicu-monitoring-system
source teleicu-monitoring-system/bin/activate
pip install -r requirements.txt
```

### 3. Clone YOLOv10 Repository

```bash
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
pip install .
```

### 4. Launch the Application

#### Windows Application

```bash
cd ..
python monitor.py
```

#### Web Application

```bash
python app.py
# Visit http://127.0.0.1:5000 in your browser
```

## 🧪 Testing

The `test.py` script provides a GUI application for testing the TeleICU Monitoring System. It's built with PyQt5 and utilizes YOLOv10 models for object and motion detection.

### Features

- Predict and display detections on images, videos, and webcam feeds
- Annotate detections with bounding boxes and labels
- Save annotated images and videos
- User-friendly interface with buttons for different prediction modes

To run the test application:

```bash
python test.py
```

## 🔄 Model Conversion

The `convert.py` script offers a simple Tkinter GUI for exporting YOLOv10 models to various formats:

- TorchScript
- ONNX
- OpenVINO
- TensorRT (GPU availability required)

To launch the conversion tool:

```bash
python convert.py
```

## 🏷️ Model Classes

Images are annotated using [Roboflow](https://roboflow.com/)

### Object Detection
- Intensivist
- Nurse
- Patient
- Family Member

### Motion Detection
- Falling
- Standing
- Sitting
- Sleeping
- Walking

## 📊 Best Model Performance

### Object Detection

YOLOv10 small model with data augmentation:

| Class         | P     | R     | mAP50 | mAP50-95 |
|---------------|-------|-------|-------|----------|
| All           | 0.771 | 0.754 | 0.794 | 0.468    |
| Family-Member | 0.821 | 0.753 | 0.796 | 0.466    |
| Intensivist   | 0.802 | 0.711 | 0.820 | 0.519    |
| Nurse         | 0.674 | 0.792 | 0.763 | 0.469    |
| Patient       | 0.788 | 0.762 | 0.795 | 0.419    |

### Motion Detection

YOLOv10 small model without data augmentation:

| Class    | P     | R     | mAP50 | mAP50-95 |
|----------|-------|-------|-------|----------|
| All      | 0.798 | 0.659 | 0.782 | 0.459    |
| Falling  | 0.554 | 0.778 | 0.755 | 0.564    |
| Sitting  | 0.903 | 0.599 | 0.798 | 0.457    |
| Sleeping | 0.944 | 0.611 | 0.883 | 0.524    |
| Standing | 0.946 | 0.658 | 0.827 | 0.449    |
| Walking  | 0.642 | 0.650 | 0.644 | 0.300    |

## 🙏 Acknowledgements

- This project is built upon the YOLOv10 model and the DeepSort algorithm.
- We extend our gratitude to the authors and contributors of the respective repositories used in this project.

## 📚 References

- [Object and Motion Dataset](https://drive.google.com/drive/folders/1HSTfpo4IAEo9k5aSaw5KK92__wk-zGVT?usp=sharing)
- [Roboflow Computer Vision Tools](https://roboflow.com/)
- [YOLOv10: Real-Time End-to-End Object Detection](https://github.com/THU-MIG/yolov10)
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- [Object and Motion Dataset](https://drive.google.com/drive/folders/1HSTfpo4IAEo9k5aSaw5KK92__wk-zGVT?usp=sharing)
- [Roboflow Computer Vision Tools](https://roboflow.com/)
- [YOLOv10: Real-Time End-to-End Object Detection](https://github.com/THU-MIG/yolov10)
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

# [TeleICU Monitoring System](https://github.com/Wydoinn/TeleICU-Monitoring-System)

- TeleICU empowers intensivists, the backbone of ICUs, to manage more patients.  This remote monitoring system leverages AI:
  - YOLOv10: This powerful deep learning model identifies patients and tracks vitals in real-time video feeds.
  - Deep SORT: It builds on YOLOv10's work, tracking patient movements to create a detailed health picture.
  
- With AI handling the heavy lifting, a single intensivist can:
  - Monitor multiple patients simultaneously, freeing up time for deeper care.
  - Intervene faster thanks to real-time monitoring, potentially improving patient outcomes.
  
TeleICU with AI is revolutionizing critical care by boosting efficiency and effectiveness.

## Demo
https://github.com/Wydoinn/TeleICU-Monitoring-System/assets/120785316/82fc5ca4-63f4-4489-8aab-ebe7e5697443

## Checkout

- Dataset - [download here](https://drive.google.com/drive/folders/1HSTfpo4IAEo9k5aSaw5KK92__wk-zGVT?usp=sharing)

- Evaluation Metrics - [click here](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/evaluation)

- Examine The Train Results. - [click here](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/examine)

- Train Notebooks - [click here](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/trains)

- Predicted Output - [click here](https://github.com/Wydoinn/TeleICU-Monitoring-System/tree/main/output)

## Installation

**1. Clone This Repository**

```
git clone https://github.com/Wydoinn/TeleICU-Monitoring-System.git
cd TeleICU-Monitoring-System
```

**2. Create New Environment**

- Using Conda

```
conda env create -f conda.yml
conda activate teleicu-monitoring-system
pip install -r requirements.txt
```

- Using Pip

```
python -m virtualenv -p python3.11.7 teleicu-monitoring-system
source teleicu-monitoring-system/bin/activate
pip install -r requirements.txt
```

**3. Clone YOLOv10 Repository**

```
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
pip install .
```

**4. TeleICU Monitoring System Application**

- For Windows Application

```
cd ..
python monitor.py
```

- For Web Application

```
python app.py
# Please visit http://127.0.0.1:5000 on your browser
```

## Test

The ```test.py``` is a GUI application for a TeleICU Monitoring System Test built with PyQt5. It utilizes YOLOv10 models for both object and motion detection. Key features include:

- Predicting and displaying detections on images, videos, and webcam feeds.
- Annotating detections with bounding boxes and labels.
- Saving annotated images and videos.
- Providing a user-friendly interface with buttons for different prediction modes.

```
python test.py
```

## Convert

```convert.py``` A simple Tkinter GUI application for exporting a YOLOv10 model to various formats including TorchScript, ONNX, OpenVINO, and TensorRT, with functionality to select the model file and handle GPU availability for TensorRT export.

```
python convert.py
```

## Model Classes

The images are annotated in [Roboflow](https://roboflow.com/)

#### Object Detection
- Intensivist
- Nurse
- Patient
- Family Member

#### Motion Detection
- Falling
- Standing
- Sitting
- Sleeping
- Walking

## Performance

#### Object Detection

YOLOv10 small model with data augmentation:

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.771 | 0.754 | 0.794 | 0.468 |
| Family-Member | 0.821 | 0.753 | 0.796 | 0.466 |
| Intensivist | 0.802 | 0.711 | 0.820 | 0.519 |
| Nurse | 0.674 | 0.792 | 0.763 | 0.469 |
| Patient | 0.788 | 0.762 | 0.795 | 0.419 |

#### Motion Detection

YOLOv10 small model without data aigmentation:

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.798 | 0.659 | 0.782 | 0.459 |
| Falling | 0.554 | 0.778 | 0.755 | 0.564 |
| Sitting | 0.903 | 0.599 | 0.798 | 0.457 |
| Sleeping | 0.944 | 0.611 | 0.883 | 0.524 |
| Standing | 0.946 | 0.658 | 0.827 | 0.449 |
| Walking | 0.642 | 0.650 | 0.644 | 0.300 |

## Acknowledgements
- This code is built upon the YOLOv10 model and the DeepSort algorithm.
- Credits to the authors and contributors of the respective repositories used in this project.

## References
- [Object and Motion Dataset](https://drive.google.com/drive/folders/1HSTfpo4IAEo9k5aSaw5KK92__wk-zGVT?usp=sharing)
- [Roboflow Computer Vision Tools](https://roboflow.com/)
- [YOLOv10: Real-Time End-to-End Object Detection](https://github.com/THU-MIG/yolov10)
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

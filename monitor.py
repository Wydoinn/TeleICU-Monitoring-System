import os
import sys
import cv2
import torch
import datetime
import numpy as np
import logging
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
)

from PyQt5.QtGui import QPixmap, QImage, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10
import supervision as sv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model path
DEFAULT_OBJECT_MODEL_PATH = "runs/object_detection/small_augmented_runs/detect/train/weights/best.pt" # Replace with different object detection model
DEFAULT_MOTION_MODEL_PATH = "runs/motion_detection/small_runs/detect/train/weights/best.pt" # Replace with different object detection model


class VideoThread(QThread):
    # Initializes the video processing object, models, trackers, and various parameters
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_source = 0  # Default to webcam (0)
        self.running = False
        self.object_model = None
        self.motion_model = None
        self.class_names = []
        self.tracker = DeepSort(max_age=20, n_init=3)
        self.colors = []
        self.class_counters = defaultdict(int)
        self.track_class_mapping = {}
        self.conf = 0.50
        self.class_id = None
        self.blur_id = None
        self.patient_class_id = 3
        self.colors = sv.ColorPalette.DEFAULT

    frame_updated = pyqtSignal(QImage)
    video_stopped = pyqtSignal()
    
    # The main loop that captures video frames, processes them, and yields the results for streaming
    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error("Error: Unable to open video source.")
            return

        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            start = datetime.datetime.now()
            tracks, object_detections, motion_detections = self.process_frame(frame)
            frame = self.draw_tracks(
                frame, tracks, object_detections, motion_detections
            )
            end = datetime.datetime.now()

            logger.info(
                f"Time to process frame {frame_count}: {(end - start).total_seconds():.2f} seconds"
            )
            frame_count += 1

            #fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}" # Uncomment to show FPS
            #cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8) # Uncomment to show FPS

            # Convert frame to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(
                rgb_image.data, w, h, ch * w, QImage.Format_RGB888
            )
            self.frame_updated.emit(qt_image)

        cap.release()
        self.video_stopped.emit()
        self.frame_updated.emit(QImage())

    # Applies object and motion detection models to a single frame, merges the results, and applies tracking
    def process_frame(self, frame):
        object_results = self.object_model(frame, verbose=False)[0]
        object_detections = sv.Detections.from_ultralytics(object_results)

        motion_results = self.motion_model(frame, verbose=False)[0]
        motion_detections = sv.Detections.from_ultralytics(motion_results)

        patient_detections = object_detections[
            object_detections.class_id == self.patient_class_id
        ]
        bounding_boxes = patient_detections.xyxy

        # Filter motion_detections to include only patients
        motion_detections = motion_detections[
            motion_detections.class_id == self.patient_class_id
        ]

        all_detections = sv.Detections.merge([object_detections, motion_detections])

        # Patient motion detection within bounding boxes
        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            motion_results = self.motion_model(roi, verbose=False)[0]
            motion_detections = sv.Detections.from_ultralytics(motion_results)
            motion_detections.xyxy[:, [0, 2]] += x1
            motion_detections.xyxy[:, [1, 3]] += y1
            all_detections = sv.Detections.merge([all_detections, motion_detections])

        detections = []

        for det in all_detections:
            class_id = det[0]
            confidence = det[1]
            bbox = det[2]

            if confidence is None:
                continue

            # Ensure bbox is a list-like structure with four values
            if isinstance(bbox, (np.float32, float, int)):
                bbox = [bbox] * 4

            x1, y1, x2, y2 = map(int, bbox)

            if self.class_id is None:
                if confidence < self.conf:
                    continue
            else:
                if class_id != self.class_id or confidence < self.conf:
                    continue

            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks, object_detections, motion_detections

    # Visualizes the tracking results on the frame, drawing bounding boxes and labels
    def draw_tracks(self, frame, tracks, object_detections, motion_detections):
        # Thickness and font parameters
        thickness = 5
        font_scale = 0.8
        font_thickness = 2

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # unique color for each class
            color = self.colors[class_id % len(self.colors)]
            
            if track_id not in self.track_class_mapping:
                self.class_counters[class_id] += 1
                self.track_class_mapping[track_id] = self.class_counters[class_id]

            text = f"{self.class_names[class_id]} {self.track_class_mapping[track_id]}"
            
            # Draw rounded rectangle with increased thickness
            cv2.ellipse(frame, (x1, y1), (5, 5), 0, 90, 180, color, thickness)
            cv2.ellipse(frame, (x2, y1), (5, 5), 0, 0, 90, color, thickness)
            cv2.ellipse(frame, (x1, y2), (5, 5), 0, 180, 270, color, thickness)
            cv2.ellipse(frame, (x2, y2), (5, 5), 0, 270, 360, color, thickness)
            cv2.line(frame, (x1 + 5, y1), (x2 - 5, y1), color, thickness)
            cv2.line(frame, (x1 + 5, y2), (x2 - 5, y2), color, thickness)
            cv2.line(frame, (x1, y1 + 5), (x1, y2 - 5), color, thickness)
            cv2.line(frame, (x2, y1 + 5), (x2, y2 - 5), color, thickness)
            
            # Add semi-transparent background for text
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width + 10, y1), color, -1)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width + 10, y1), color, thickness)
            
            # Add text with improved positioning and larger size
            cv2.putText(frame, text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            if self.blur_id is not None and class_id == self.blur_id:
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        # Object model annotation
        bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=thickness)
        object_label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.DEFAULT,
            text_position=sv.Position.TOP_LEFT,
            text_scale=font_scale,
            text_thickness=font_thickness,
            text_padding=5
        )
        frame = bounding_box_annotator.annotate(scene=frame, detections=object_detections)
        frame = object_label_annotator.annotate(scene=frame, detections=object_detections)

        # Motion model annotation
        motion_label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.DEFAULT,
            text_position=sv.Position.TOP_RIGHT,
            text_scale=font_scale,
            text_thickness=font_thickness,
            text_padding=5
        )
        frame = motion_label_annotator.annotate(scene=frame, detections=motion_detections)

        return frame

    # Loads the object detection and motion detection models, and initializes class names and colors
    def load_models(self, object_model_path, motion_model_path):
        self.object_model = self.load_model(object_model_path)
        self.motion_model = self.load_model(motion_model_path)
        self.class_names = self.load_class_names()
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))

    # Loads a specific YOLO model and sets up the appropriate processing device (CUDA, MPS, or CPU)
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"Model weights not found at {model_path}")
            raise FileNotFoundError("Model weights file not found")

        model = YOLOv10(model_path)

        # Use CUDA or CPU or MPS
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model.to(device)
        logger.info(f"Using {device} as processing device for {model_path}")
        return model

    # Loads the names of object classes from a file for labeling detected objects
    def load_class_names(self):
        classes_path = "coco.names"
        if not os.path.exists(classes_path):
            logger.error(f"Class names file not found at {classes_path}")
            raise FileNotFoundError("Class names file not found")

        with open(classes_path, "r") as f:
            class_names = f.read().strip().split("\n")
        return class_names


class App(QMainWindow):
    # Sets up the main window, layout, and UI components and creates a VideoThread instance and connects its signals
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TeleICU Monitoring System")
        self.setFixedSize(1080, 800)  # Main window resolution

        # Floral color palette
        palette = QPalette()
        self.setPalette(palette)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create a label to display the video
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label)

        # Create buttons
        self.buttons_layout = QHBoxLayout()
        self.layout.addLayout(self.buttons_layout)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_video)
        self.buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_video)
        self.buttons_layout.addWidget(self.stop_button)

        # Create a thread for video processing
        self.thread = VideoThread(self)
        self.thread.frame_updated.connect(self.update_video_label)
        self.thread.video_stopped.connect(self.on_video_stopped)

        # Load default models on startup
        self.thread.load_models(DEFAULT_OBJECT_MODEL_PATH, DEFAULT_MOTION_MODEL_PATH)

    # Starts the video processing thread
    def start_video(self):
        self.thread.running = True
        self.thread.start()

    # Stops the video processing thread
    def stop_video(self):
        self.thread.running = False

    @pyqtSlot(QImage)
    # Updates the GUI with the latest processed frame
    def update_video_label(self, image):
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(
                pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
            )
        else:
            self.video_label.clear()

    @pyqtSlot()
    # Clears the video label when the video stops
    def on_video_stopped(self):
        self.video_label.clear()
        self.video_label.setStyleSheet("background-color: black;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    desktop = QApplication.desktop().screenGeometry()
    window.setGeometry((desktop.width() - window.width()) // 2, 
                       (desktop.height() - window.height()) // 2, 
                       window.width(), window.height())
    window.show()
    sys.exit(app.exec_())

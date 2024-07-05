import sys
import os
import datetime
import cv2
import supervision as sv
from ultralytics import YOLOv10
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox

# Load the models
object_model = YOLOv10("runs/object_detection/small_augmented_runs/detect/train/weights/best.pt")  # Replace with different object detection model
motion_model = YOLOv10("runs/motion_detection/small_runs/detect/train/weights/best.pt")  # Replace with different motion detection model

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TeleICU Monitoring System Test")
        self.setFixedSize(1080, 840)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c2f33;
            }
        """)

        screen_geometry = QApplication.desktop().screenGeometry()
        window_geometry = self.geometry()
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        self.move(x, y)
        
        self.image_label = QLabel()
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: black;
                color: white;
                font-size: 16px;
                padding: 10px;
            }
        """)

        predict_image_button = QPushButton("Predict on Image")
        predict_image_button.setStyleSheet("""
            QPushButton {
                background-color: #5e81ac;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #81a1c1;
            }
        """)
        predict_image_button.clicked.connect(self.predict_image)

        predict_video_button = QPushButton("Predict on Video")
        predict_video_button.setStyleSheet("""
            QPushButton {
                background-color: #5e81ac;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #81a1c1;
            }
        """)
        self.video_writer = None
        predict_video_button.clicked.connect(self.predict_video)

        predict_webcam_button = QPushButton("Predict on Webcam")
        predict_webcam_button.setStyleSheet("""
            QPushButton {
                background-color: #5e81ac;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #81a1c1;
            }
        """)
        predict_webcam_button.clicked.connect(self.predict_webcam)

        quit_button = QPushButton("Quit")
        quit_button.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e57373;
            }
        """)
        quit_button.clicked.connect(self.quit_app)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(predict_image_button)
        layout.addWidget(predict_video_button)
        layout.addWidget(predict_webcam_button)
        layout.addWidget(quit_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.video_capture = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)

    # Handles image prediction
    def predict_image(self):
        self.video_timer.stop()

        if self.video_capture:
            self.video_capture.release()
            cv2.destroyAllWindows()

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")

        if file_path:
            image = cv2.imread(file_path)
            results = object_model(source=image, conf=0.25)[0]

            detections = sv.Detections.from_ultralytics(results)
            
            thickness = 5
            font_scale = 0.8
            font_thickness = 2
            
            bounding_box_annotator = sv.BoundingBoxAnnotator(
                color=sv.ColorPalette.DEFAULT, thickness=thickness
            )
            label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette.DEFAULT,
                text_position=sv.Position.TOP_LEFT,
                text_scale=font_scale,
                text_thickness=font_thickness,
                text_padding=5, 
            )

            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/images/annotated_image_{current_time}.jpg"

            os.makedirs("output/images", exist_ok=True)
            cv2.imwrite(filename, annotated_image)
                
            pixmap = QPixmap(filename)
            self.image_label.setPixmap(pixmap)
            self.center_pixmap_in_label(pixmap)

    # Sets up video prediction
    def predict_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")

        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)

            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            os.makedirs("output/videos", exist_ok=True)

            input_file_name = os.path.basename(file_path)
            input_file_name_without_ext, _ = os.path.splitext(input_file_name)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_name = f"{input_file_name_without_ext}_{current_time}_annotated.mp4"
            output_file_path = os.path.join("output/videos", output_file_name)

            self.video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
            self.video_timer.start(25)

    # Sets up webcam prediction
    def predict_webcam(self):
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
            cv2.destroyAllWindows()

        self.video_capture = cv2.VideoCapture(0)

        if self.video_capture.isOpened():
            self.video_timer.start(25)
        else:
            QMessageBox.warning(self, "Error", "Failed to open the webcam.")

    # Processes each video or webcam frame
    def update_video_frame(self):
        ret, frame = self.video_capture.read()

        if ret:
            # Object detection
            object_results = object_model(frame)[0]
            object_detections = sv.Detections.from_ultralytics(object_results)

            # Filter for Patient class
            target_class_id = 3  # Patient's class ID is 3
            target_detections = object_detections[object_detections.class_id == target_class_id]
            bounding_boxes = target_detections.xyxy

            # Motion detection
            motion_detections = sv.Detections.empty()
            for box in bounding_boxes:
                x1, y1, x2, y2 = box.astype(int)
                roi = frame[y1:y2, x1:x2]
                motion_results = motion_model(roi)[0]
                motion_detections_roi = sv.Detections.from_ultralytics(motion_results)
                motion_detections_roi.xyxy[:, [0, 2]] += x1
                motion_detections_roi.xyxy[:, [1, 3]] += y1
                motion_detections = sv.Detections.merge([motion_detections, motion_detections_roi])

            # Combine object detections and motion detections
            all_detections = sv.Detections.merge([object_detections, motion_detections])

            # Custom annotation appearance
            thickness = 4
            font_scale = 0.8
            font_thickness = 2

            bounding_box_annotator = sv.BoundingBoxAnnotator(
                color=sv.ColorPalette.DEFAULT, thickness=thickness
            )
            object_label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette.DEFAULT,
                text_position=sv.Position.TOP_LEFT,
                text_scale=font_scale,
                text_thickness=font_thickness,
                text_padding=5,
            )
            motion_label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette.DEFAULT,
                text_position=sv.Position.TOP_RIGHT,
                text_scale=font_scale,
                text_thickness=font_thickness,
                text_padding=5,
            )

            frame = bounding_box_annotator.annotate(scene=frame, detections=all_detections)
            frame = object_label_annotator.annotate(scene=frame, detections=object_detections)
            frame = motion_label_annotator.annotate(scene=frame, detections=motion_detections)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.center_pixmap_in_label(pixmap)

        else:
            self.video_timer.stop()
            self.video_capture.release()
            cv2.destroyAllWindows()

    # Centers the display
    def center_pixmap_in_label(self, pixmap):
        label_size = self.image_label.size()

        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio)

        pixmap_size = scaled_pixmap.size()

        x = (label_size.width() - pixmap_size.width()) // 2
        y = (label_size.height() - pixmap_size.height()) // 2

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setContentsMargins(x, y, 0, 0)

    # Handles the application quit process
    def quit_app(self):
        reply = QMessageBox.question(self, "Quit", "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.video_capture:
                self.video_capture.release()
                cv2.destroyAllWindows()
            QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

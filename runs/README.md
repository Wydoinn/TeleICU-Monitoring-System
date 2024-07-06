# Train Results

After training a YOLO model, we'll find a folder named "runs" created by default. This folder stores important information about your training process.

**Here's what's inside the "runs" folder:**

- Training logs: These files track details like loss, accuracy, and other metrics throughout the training process. They're helpful for analyzing training progress and identifying potential issues.

- Weights: The trained model weights are typically saved within the "runs" folder, often under a subfolder named "train/weights". The best performing weights might have a filename like "best.pt". These weights are crucial for using your trained model for object detection on new images.

- Optional outputs: Depending on your YOLO version and training settings, you might also find additional outputs in the "runs" folder. These could include visualizations of training progress (like "result.png") or confusion matrices.

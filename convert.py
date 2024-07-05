import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLOv10
import torch

window = tk.Tk()
window.title("Model Export")
window.geometry("250x100")

# Create a frame for the buttons
frame = ttk.Frame(window, padding=10)
frame.grid()

# Handle model selection
def model():
    file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Model", "*.pt")])
    if file_path:
        model = YOLOv10(file_path)
        return model
    else:
        return None

# Handle button clicks for TensorRT
def export(format):
    model = model()
    if model:
        print(f"Exporting model in {format} format...")
        if format == 'engine':
            # Check if a GPU is available to exposrt the TensorRT format
            if not torch.cuda.is_available():
                warning("Warning: TensorRT requires a GPU. Exporting to TensorRT will not work on this system.")
        model.export(format=format)
    else:
        print("No model selected.")

# Show a warning message box
def warning(message):
    messagebox.showwarning("Warning", message)

# Buttons for each format
torchscript_button = ttk.Button(frame, text="TorchScript", command=lambda: export('torchscript'))
torchscript_button.grid(row=0, column=0, padx=5, pady=5)

onnx_button = ttk.Button(frame, text="ONNX", command=lambda: export('onnx'))
onnx_button.grid(row=0, column=1, padx=5, pady=5)

openvino_button = ttk.Button(frame, text="OpenVINO", command=lambda: export('openvino'))
openvino_button.grid(row=1, column=0, padx=5, pady=5)

tftrt_button = ttk.Button(frame, text="TensorRT", command=lambda: export('engine'))
tftrt_button.grid(row=1, column=1, padx=5, pady=5)

window.mainloop()

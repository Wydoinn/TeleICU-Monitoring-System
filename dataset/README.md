# Dataset

There are two datasets: one for object detection and another for motion detection. Both datasets are divided into two categories: one with and one without data augmentation. 

- Object Detection (data augmentation) - 11440  Images
- Motion Detection (data augmentation) - 21547 Images

- Object Detection (no data augmentation) - 3000  Images
- Motion Detection (no data augmentation) - 5697 Images

## 📥 Dataset Acquisition

### Required Classes

#### Object Detection (4 classes)
- `Family-Member` - Family members visiting ICU
- `Intensivist` - Medical specialists in intensive care
- `Nurse` - Healthcare nursing staff
- `Patient` - ICU patients

#### Motion Detection (5 classes)
- `Falling` - Person falling down
- `Sitting` - Person in sitting position
- `Sleeping` - Person sleeping/lying down
- `Standing` - Person standing upright
- `Walking` - Person walking

### Alternative Dataset Sources

Since the original dataset may not be accessible, here are alternative approaches to obtain datasets:

#### 1. **Roboflow Universe**
The project uses Roboflow for annotation. Search for similar datasets:
- Visit [Roboflow Universe](https://universe.roboflow.com/)
- Search for: "hospital", "healthcare", "patient monitoring", "fall detection", "human activity"
- Look for datasets with similar classes that can be adapted
- Many datasets are available with open licenses

#### 2. **Public Datasets for Base Training**

**For Object Detection:**
- **COCO Dataset** - Contains person class, can be fine-tuned
- **Open Images Dataset** - Healthcare and person-related images
- Create custom dataset by collecting images from:
  - Public domain medical education videos (with proper licensing)
  - Stock photo websites with healthcare themes
  - Simulated ICU environments

**For Motion Detection:**
- **UP-Fall Detection Dataset** - Fall detection research dataset
- **NTU RGB+D** - Human activity recognition dataset
- **Kinetics Dataset** - Human action videos
- **UCF101** - Action recognition dataset containing sitting, standing, walking activities

#### 3. **Creating Your Own Dataset**

If public datasets don't meet your needs, collect and annotate your own:

**Data Collection:**
1. Record videos in simulated ICU environments
2. Extract frames from videos
3. Ensure diverse lighting, angles, and scenarios
4. Obtain proper consent and follow ethical guidelines for healthcare data

**Annotation:**
1. Use [Roboflow](https://roboflow.com/) for labeling
2. Follow YOLO format for bounding boxes
3. Ensure balanced class distribution
4. Include edge cases (occlusions, multiple people, various poses)

**Tools:**
- [Roboflow](https://roboflow.com/) - Recommended (as used in this project)
- [LabelImg](https://github.com/tzutalin/labelImg)
- [CVAT](https://github.com/opencv/cvat)
- [Labelbox](https://labelbox.com/)

### Dataset Format

Datasets should be in **YOLO format**:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label file should contain:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where coordinates are normalized (0-1) relative to image dimensions.

## 📊 Dataset Specifications

### The Dataset with Data Augmentation

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 5 versions of each source image:
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random brightness adjustment of between -20 and +20 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 1.5 percent of pixels.

### The Dataset with No Data Augmentation

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

No image augmentation techniques were applied.

## 🔗 Useful Resources

- [Roboflow Universe - Browse Datasets](https://universe.roboflow.com/)
- [YOLOv10 Training Documentation](https://github.com/THU-MIG/yolov10)
- [Data Augmentation Techniques](https://roboflow.com/augment)
- [Healthcare AI Dataset Guidelines](https://www.nature.com/articles/s41591-020-01205-0)

## 📧 Contact

For dataset-related inquiries or collaboration on dataset collection, please open an issue in this repository.

## ⚖️ Ethical Considerations

When collecting or using healthcare-related datasets:
- Ensure patient privacy and HIPAA compliance
- Obtain necessary consent and approvals
- Use synthetic or simulated data when possible
- Follow institutional review board (IRB) guidelines
- Anonymize all personal information

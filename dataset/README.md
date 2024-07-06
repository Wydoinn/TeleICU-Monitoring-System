# Dataset

There are two datasets: one for object detection and another for motion detection. Both datasets are divided into two categories: one with and one without data augmentation. 

- Object Detection (data augmentation) - 11440  Images
- Motion Detection (data augmentation) - 21547 Images

- Object Detection (no data augmentation) - 3000  Images
- Motion Detection (no data augmentation) - 5697 Images

## The Dataset with Data Augmentation

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 5 versions of each source image:
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random brigthness adjustment of between -20 and +20 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 1.5 percent of pixels.

## The Dataset with No Data Augmentation

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

No image augmentation techniques were applied.

Dataset Link - [Download Here](https://drive.google.com/drive/folders/1HSTfpo4IAEo9k5aSaw5KK92__wk-zGVT?usp=sharing)

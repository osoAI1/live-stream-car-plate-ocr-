# The Dataset:  [Link of data](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
# This data not pre_trained: to train it use yolo command to get -----> "license_plate_detector.pt"
# Resize: Stretch to 640x640
# Tha Dataset have 4 VERSIONS


# last VERSIONS 4 :Name: resized640_aug3x-ACCURATE:
 Augmentations:
   Flip: Horizontal
   Crop: 0% Minimum Zoom, 15% Maximum Zoom
   Rotation: Between -10° and +10°
   Shear: ±2° Horizontal, ±2° Vertical
   Grayscale: Apply to 10% of images
   Hue: Between -15° and +15°
   Saturation: Between -15% and +15%
   Brightness: Between -15% and +15%
   Exposure: Between -15% and +15%
   Blur: Up to 0.5px
   Cutout: 5 boxes with 2% size each
# Total: 24242 Images


# VERSIONS 3:Name: resized640_aug3x-FAST:
Augmentations:
 Flip: Horizontal
 Crop: 0% Minimum Zoom, 15% Maximum Zoom
 Rotation: Between -10° and +10°
 Shear: ±2° Horizontal, ±2° Vertical
 Grayscale: Apply to 10% of images
 Hue: Between -15° and +15°
 Saturation: Between -15% and +15%
 Brightness: Between -15% and +15%
 Exposure: Between -15% and +15%
 Blur: Up to 0.5px
 Cutout: 5 boxes with 2% size each
# Total: 24242 Images

# VERSIONS 2:Name: resized640_noAugmentation-FAST: 10126 Total Images

# VERSIONS 1:Name: raw-images: 10126 Total Images


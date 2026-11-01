# Technical Project Report
## Vision-Based Robot Navigation Using YOLO11n

### Abstract

This project implements a vision-based obstacle detection and basic navigation-decision pipeline using the YOLO11n object detection model. A pretrained YOLO11n model was fine-tuned on a prepared version of the ROD obstacle detection dataset. Predictions were analyzed spatially to demonstrate simple navigation decisions.

### 1. Problem Statement

A mobile robot operating in an environment needs to perceive objects and obstacles from camera images. Object detection can provide object classes and image locations that can support navigation decisions. This project develops an object detection and image-zone analysis prototype.

### 2. Dataset

The ROD obstacle detection dataset was obtained from Kaggle.

- Total images: 24,326
- Training images: 19,186
- Validation images: 3,511
- Test images: 1,629
- Classes: 25
- Final converted object boxes: 40,195
- Empty background label files: 99

The source annotations mixed standard YOLO detection boxes and polygon segmentation labels. A conversion procedure transformed polygons into axis-aligned bounding boxes. The original annotation files were preserved. A maximum normalized boundary overflow of 0.000010416667 was observed in 190 boxes and clipped to the valid image range.

### 3. Methodology

The workflow consisted of:

1. Downloading and extracting the dataset.
2. Inspecting image and label directories.
3. Validating annotation class IDs and coordinate ranges.
4. Converting polygon segmentation annotations to bounding boxes.
5. Creating a physically copied dataset with matching image and label directories.
6. Loading pretrained YOLO11n weights.
7. Training the model on a subset of the training split.
8. Evaluating against validation and independent test splits.
9. Performing sample image inference.
10. Applying a three-zone navigation decision heuristic.

### 4. Model Architecture

YOLO11n was selected because it is a lightweight object detection model suitable for real-time perception experiments. Pretrained weights were used as the initialization, and the detection head was adapted to the 25 dataset classes.

### 5. Training Configuration

| Parameter | Setting |
|---|---|
| Model | YOLO11n |
| Epochs | 5 |
| Training fraction | 0.20 |
| Images used for training | 3,837 |
| Validation images | 3,511 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | Automatically selected AdamW |
| Hardware | NVIDIA Tesla T4 |
| Mixed precision | Enabled |
| Random seed | 42 |

The five epochs completed in approximately 0.111 hours according to the training log.

### 6. Evaluation Metrics

Precision measures the proportion of predicted detections that are correct at the evaluation operating point.

Recall measures the proportion of ground-truth objects that were detected.

mAP@50 computes mean Average Precision at an IoU threshold of 0.50.

mAP@50-95 averages mAP across IoU thresholds from 0.50 to 0.95.

### 7. Validation Results

| Metric | Result |
|---|---:|
| Images | 3,511 |
| Instances | 5,572 |
| Precision | 0.7920 |
| Recall | 0.3480 |
| mAP@50 | 0.3910 |
| mAP@50-95 | 0.2760 |

### 8. Independent Test Results

The test split was evaluated separately after training.

| Metric | Result |
|---|---:|
| Images | 1,629 |
| Instances | 2,372 |
| Precision | 0.7724 |
| Recall | 0.3365 |
| mAP@50 | 0.4006 |
| mAP@50-95 | 0.2860 |

The test performance is similar to the validation performance. Recall remains low, indicating that many ground-truth objects were not detected. Performance also varies substantially by class.

### 9. Inference Results

Eight test images were used in the demonstration. The model generated 12 object detections in total at a confidence threshold of 0.25.

Annotated images include object class labels, bounding boxes and confidence scores.

The reported test evaluation timing on the Tesla T4 was approximately 1.2 ms preprocessing, 4.8 ms inference and 1.5 ms postprocessing per image.

### 10. Navigation Decision Demonstration

The image was divided into three horizontal zones: left, center and right.

For each detection, a risk contribution was calculated from confidence and normalized bounding-box area. The contribution was distributed across zones according to bounding-box overlap.

The final heuristic used a center-risk threshold of 0.10 and a side-risk tie margin of 0.03.

For eight sample images, the resulting decisions were:

| Decision | Count |
|---|---:|
| FORWARD | 1 |
| TURN LEFT | 0 |
| TURN RIGHT | 0 |
| STOP | 7 |

These are demonstration outputs only. The dataset does not contain ground-truth robot navigation actions, and therefore no navigation accuracy or collision-avoidance success rate is claimed.

### 11. Limitations

1. The baseline was trained for only five epochs using 20% of the training images.
2. Recall is low, so the model misses many objects.
3. The dataset has class imbalance.
4. Polygon-to-box conversion removes contour information.
5. A bounding box is not a physical distance measurement.
6. The navigation rule is heuristic and does not establish a collision-free path.
7. The project has not been deployed on a physical robot.

### 12. Conclusion

A working YOLO11n obstacle-detection baseline and an image-based navigation-decision demonstration were implemented. Dataset preparation, annotation conversion, independent test evaluation and visual result generation were completed. The experiment demonstrates the use of deep learning for robotic visual perception and provides a foundation for future improvements such as depth estimation, tracking, temporal decision-making and ROS 2 integration.

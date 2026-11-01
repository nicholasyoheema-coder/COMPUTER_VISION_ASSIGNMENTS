# Project Report
## Vision-Based Robotic Pick-and-Place Using YOLO11n and Object Localization

### 1. Abstract

This project implements a vision-based detection and virtual sorting workflow for bottles and cups. A pretrained YOLO11n model detects the two object categories. Each detected bounding box is converted into an estimated image-space pick point using its centre. Normalized coordinates are used to create a virtual sorting plan.

The experiment evaluated 89 annotated images containing 117 objects. The model produced 94 detections and achieved an overall precision of 0.8085, recall of 0.6496, and F1-score of 0.7204 at confidence 0.25 and IoU 0.50.

### 2. Objectives

- Detect bottles and cups in images.
- Estimate 2D object centres.
- Generate normalized image coordinates.
- Evaluate detections against XML ground-truth annotations.
- Generate a class-based virtual sorting plan.

### 3. Dataset

The Kaggle Bottles and Cups Dataset contained 100 images. Eighty-nine images had XML annotations and were used for evaluation. Those annotations contained 97 bottles and 20 cups, for a total of 117 objects. Eleven images without XML annotations were excluded.

The dataset metadata identified its license as `copyright-authors`; original images and annotation files are not redistributed with this project.

### 4. Methodology

The pretrained YOLO11n model was loaded with COCO weights. Bottle and cup class IDs were verified as 39 and 41 respectively. Detection was run at image size 640 with confidence threshold 0.25.

For a bounding box with corners (x1, y1) and (x2, y2), its centre is:

- cx = (x1 + x2) / 2
- cy = (y1 + y2) / 2
- normalized_x = cx / image_width
- normalized_y = cy / image_height

Predictions were matched to ground-truth objects of the same class in the same image. A prediction was a true positive when IoU was at least 0.50. Unmatched predictions were false positives, and unmatched ground-truth objects were false negatives. All annotated images were included, even when they had no predicted objects.

### 5. Results

| Metric | Value |
|---|---:|
| Images evaluated | 89 |
| Ground-truth objects | 117 |
| Predicted detections | 94 |
| True positives | 76 |
| False positives | 18 |
| False negatives | 41 |
| Precision | 0.8085 |
| Recall | 0.6496 |
| F1-score | 0.7204 |

#### Class-wise results

| bottle | 97 | 60 | 3 | 37 | 0.9524 | 0.6186 | 0.7500 |
| cup | 20 | 16 | 15 | 4 | 0.5161 | 0.8000 | 0.6275 |

Matched detections had mean IoU values of approximately 0.9297 for bottles and 0.9191 for cups. These values describe bounding-box overlap, not physical grasp accuracy.

### 6. Virtual sorting

The simulation generated 94 operations from the 94 detections:
- Bottle destination assignments: 63
- Cup destination assignments: 31

The virtual destination points are illustrative normalized coordinates. No real robot movement was performed.

### 7. Discussion

Bottle detections achieved higher precision than cup detections. The cup category produced more false positives. Bottle recall was lower, indicating that some annotated bottles were missed. These missed detections and false detections could cause incorrect sorting without additional verification.

### 8. Limitations

The model was not fine-tuned on this dataset. The benchmark is small and imbalanced. The pick points are 2D bounding-box centres and are not calibrated physical coordinates. The project does not implement camera calibration, depth estimation, grasp planning, inverse kinematics, collision checking or physical robot control.

### 9. Future scope

Future work can include a larger licensed dataset, detector fine-tuning, camera calibration, depth estimation, robot-frame coordinate transformation, grasp planning and integration with a robot-arm simulator.

### 10. Conclusion

The project demonstrates pretrained object detection, image-space localization, quantitative evaluation against XML ground truth and class-based virtual sorting. Its results describe a software-based computer-vision prototype, not physical robotic manipulation.

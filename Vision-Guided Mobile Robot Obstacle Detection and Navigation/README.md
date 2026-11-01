# Vision-Based Robot Navigation Using YOLO11n

## 1. Problem Statement

Autonomous mobile robots require visual perception to identify obstacles in their surroundings and make navigation decisions. This project develops a camera-based obstacle detection system using YOLO11n and a lightweight image-zone decision heuristic.

## 2. Objectives

- Detect obstacles and relevant objects in camera images.
- Localize detections using bounding boxes.
- Count detected objects by class.
- Divide the image into left, center and right navigation zones.
- Generate basic FORWARD, TURN LEFT, TURN RIGHT and STOP decisions.
- Evaluate object detection using precision, recall and mean Average Precision.

## 3. Dataset

**ROD - Obstacle Detection Dataset**

Source: [Kaggle dataset](https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset)

The dataset contains 24,326 images, 25 classes and 40,195 final object annotations.

The original annotations mixed detection boxes and polygon segmentation labels. Polygon labels were converted into enclosing axis-aligned bounding boxes. Small measured boundary overflows were clipped. Original downloaded data was preserved during experimentation.

The dataset itself is not redistributed in this repository.

## 4. Methodology

1. Dataset download and inspection.
2. Annotation-format validation.
3. Polygon-to-bounding-box conversion.
4. Dataset preparation and path verification.
5. Loading pretrained YOLO11n weights.
6. Fine-tuning on a 20% training subset for five epochs.
7. Validation and independent test evaluation.
8. Object detection and object counting.
9. Three-zone spatial risk analysis.
10. Basic navigation decision generation.

### Navigation logic

Detected bounding boxes contribute to normalized LEFT, CENTER and RIGHT risk scores according to their overlap with each image zone.

The heuristic uses a center-risk threshold of 0.10 and a side-risk tie margin of 0.03. It is an image-based demonstration and is not a physical distance measurement or a verified collision-free planner.

## 5. Model and Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Pretrained weights | yolo11n.pt |
| Epochs | 5 |
| Training fraction | 20% |
| Training images used | 3,837 |
| Validation images | 3,511 |
| Input size | 640 x 640 |
| Batch size | 16 |
| Optimizer | Automatically selected AdamW |
| GPU | NVIDIA Tesla T4 |
| Number of classes | 25 |

This is a short baseline experiment, not a fully optimized model.

## 6. Actual Results

### Validation results

| Metric | Result |
|---|---:|
| Precision | 0.7920 |
| Recall | 0.3480 |
| mAP@50 | 0.3910 |
| mAP@50-95 | 0.2760 |

### Independent test results

| Metric | Result |
|---|---:|
| Precision | 0.7724 |
| Recall | 0.3365 |
| mAP@50 | 0.4006 |
| mAP@50-95 | 0.2860 |

The independent test evaluation used 1,629 images containing 2,372 annotated instances.

### Inference speed

On the NVIDIA Tesla T4, Ultralytics reported approximately:

- Preprocessing: 1.2 ms/image
- Inference: 4.8 ms/image
- Postprocessing: 1.5 ms/image

These are model evaluation timings, not a full physical robot system's end-to-end latency.

### Prediction demonstration

Eight independent test images were used for the sample demonstration. The detector produced 12 detections in total at confidence threshold 0.25.

### Navigation demonstration

| Decision | Images |
|---|---:|
| FORWARD | 1 |
| TURN LEFT | 0 |
| TURN RIGHT | 0 |
| STOP | 7 |

These are outputs from the heuristic on eight sample images. They are not navigation accuracy measurements.

## 7. Tools and Libraries

- Python
- Google Colab
- Ultralytics YOLO11n
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PyYAML

## 8. Repository Contents

- `src/prepare_dataset.py`: Converts mixed YOLO annotations to detection format.
- `src/train.py`: Reproduces the baseline training experiment.
- `src/evaluate.py`: Runs validation or independent test evaluation.
- `src/predict.py`: Detects objects, counts classes and generates a navigation heuristic output.
- `models/best.pt`: Trained YOLO11n weights.
- `results/`: Metrics, curves, confusion matrices and sample outputs.
- `docs/project_report.md`: Technical project report.

## 9. Installation and Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and extract the ROD dataset from Kaggle.

Prepare the dataset:

```bash
python src/prepare_dataset.py --source /path/to/extracted/dataset --output data/rod_detection_dataset_fixed
```

Train:

```bash
python src/train.py --data data/rod_detection_dataset_fixed/data.yaml
```

Evaluate on the independent test split:

```bash
python src/evaluate.py --weights models/best.pt --data data/rod_detection_dataset_fixed/data.yaml --split test
```

Run inference on an image:

```bash
python src/predict.py --source path/to/image.jpg --weights models/best.pt
```

The prediction script saves an annotated image and a JSON summary containing detected objects, zone risk scores and the heuristic navigation decision.

## 10. Limitations

- Training used only 20% of the training images and five epochs.
- Recall and mAP indicate that further training and optimization are needed.
- Performance differs between object classes.
- The dataset has class imbalance.
- Polygon-to-box conversion loses object contour information.
- The navigation heuristic uses 2D bounding boxes and does not measure physical distance.
- Navigation actions have no ground-truth labels, so navigation accuracy cannot be calculated.
- The system has not been validated on a physical robot.

## 11. Conclusion

The project demonstrates a complete image-based perception pipeline using YOLO11n, from obstacle detection and counting to a simple spatial navigation decision heuristic. The trained model was evaluated on a separate test split, and the results, visualizations and source code are included for reproducibility.

The experiment provides a working baseline for future integration with camera calibration, depth sensing, temporal obstacle tracking and ROS 2 robot navigation.

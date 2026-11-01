# Vision-Based Robotic Pick-and-Place Using YOLO11n

A computer-vision project for detecting bottles and cups, estimating 2D image-space pick points, and generating a virtual class-based sorting plan.

## Method

1. Load a pretrained YOLO11n COCO model.
2. Detect bottles and cups.
3. Extract bounding-box coordinates.
4. Calculate each bounding-box centre as an estimated 2D pick point.
5. Normalize the centre coordinates relative to image width and height.
6. Assign detected objects to a virtual bottle or cup destination bin.

**This is a software-based 2D localization and sorting demonstration. It does not control a physical robot or calculate calibrated 3D grasp coordinates.**

## Experimental results

| Metric | Overall result |
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

### Class-wise results

| Class | Ground truth | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 97 | 60 | 3 | 37 | 0.9524 | 0.6186 | 0.7500 |
| cup | 20 | 16 | 15 | 4 | 0.5161 | 0.8000 | 0.6275 |

Evaluation used confidence threshold **0.25** and IoU threshold **0.50**. All 89 annotated images were included, including images where no detections were found. These are threshold-based precision, recall and F1 metrics, not mAP.

## Pick-and-place simulation

- Bottle destination: normalized virtual coordinate (0.25, 0.50)
- Cup destination: normalized virtual coordinate (0.75, 0.50)
- Operations generated: 94

The destination coordinates are illustrative values for the sorting logic, not calibrated robot coordinates.

## Dataset

The experiment used the [Bottles and Cups Dataset on Kaggle](https://www.kaggle.com/datasets/dataclusterlabs/bottles-and-cups-dataset).

- 100 images available
- 89 XML-annotated images evaluated
- 117 annotated objects: 97 bottles and 20 cups

The dataset metadata identifies the license as `copyright-authors`. Original images and XML annotations are therefore not included in this repository. Download the dataset separately and follow its licensing terms. See `dataset/README.md`.

## Installation

Install dependencies using:

`pip install -r requirements.txt`

The pretrained `yolo11n.pt` checkpoint is downloaded automatically by Ultralytics if it is not already available.

## Run the project

Detection and localization:

`python src/detect_and_localize.py --dataset /path/to/dataset`

Evaluation:

`python src/evaluate.py --dataset /path/to/dataset --predictions results/detections/pick_point_detections.csv`

Pick-and-place simulation:

`python src/simulate_pick_place.py --predictions results/detections/pick_point_detections.csv`

## Repository structure

- `src/` — detection, evaluation and sorting scripts
- `dataset/` — dataset download and license instructions
- `models/` — model weight instructions
- `docs/` — project report and experiment notes
- `results/detections/` — prediction and localization CSV
- `results/metrics/` — corrected evaluation data
- `results/simulation/` — virtual sorting plan
- `results/figures/` — aggregate performance charts

## Limitations

- The detector uses pretrained COCO weights; it was not fine-tuned on this dataset.
- There are only 89 annotated images and the classes are imbalanced.
- Cup detections have lower precision than bottle detections.
- Bounding-box centres are approximate 2D pick points, not validated grasp points.
- Camera calibration, depth estimation, inverse kinematics, collision avoidance, grasp planning and physical robot execution are not implemented.

## Future improvements

- Collect a larger and more balanced, appropriately licensed dataset.
- Fine-tune the detector on suitable training data.
- Add confidence and temporal verification.
- Calibrate the camera and estimate object depth.
- Transform image coordinates into robot workspace coordinates.
- Integrate robotic grasp planning and a robot-arm simulator.

## Conclusion

The project demonstrates pretrained object detection, 2D object localization, quantitative comparison with XML annotations, and class-based virtual sorting.

# Experiment Notes

- Model: YOLO11n pretrained COCO weights
- Ultralytics version: 8.4.197
- Inference hardware: NVIDIA Tesla T4
- Input size: 640
- Confidence threshold: 0.25
- IoU threshold: 0.50
- Custom training: none

## Dataset audit

- Total images: 100
- Annotated images evaluated: 89
- Bottle annotations: 97
- Cup annotations: 20
- Ground-truth objects: 117

## Detection

- Total detections: 94
- Bottle detections: 63
- Cup detections: 31
- Mean confidence: 0.6314

## Corrected evaluation

- TP: 76
- FP: 18
- FN: 41
- Precision: 0.8085
- Recall: 0.6496
- F1-score: 0.7204

All 89 annotated images were included, so images without predictions contribute their missed objects to false negatives. These are threshold-based metrics, not mAP.

## Simulation

- Planned operations: 94
- Bottle bin assignments: 63
- Cup bin assignments: 31

Pick points are bounding-box centres in 2D image coordinates, not calibrated physical coordinates.

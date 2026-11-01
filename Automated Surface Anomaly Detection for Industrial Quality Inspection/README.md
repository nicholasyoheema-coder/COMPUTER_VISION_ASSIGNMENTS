# Industrial Surface Defect Detection Using YOLO11n

Custom-trained YOLO11n detector for six types of steel surface defects.

## Defect classes
0. Crazing
1. Inclusion
2. Patches
3. Pitted surface
4. Rolled-in scale
5. Scratches

## Dataset
NEU Surface Defect Database — 1,800 images and 4,189 annotated objects.
Training: 1,152 images. Internal validation: 288 images.
Final independent test: 360 images containing 854 instances.

One XML file (crazing_240.xml) was stored in the opposite split folder.
It was matched globally by filename and assigned according to its image.

The dataset listing reports an unknown licence. Original images and XML
annotations are not included. Obtain the dataset separately from Kaggle:
https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

## Training configuration
- Model: YOLO11n, pretrained COCO weights
- Custom training: Yes
- Epochs: 30
- Image size: 320
- Batch size: 16
- GPU: NVIDIA Tesla T4
- Ultralytics: 8.4.197

## Final independent test results
| Metric | Result |
|---|---:|
| Precision | 0.5924 |
| Recall | 0.7081 |
| mAP50 | 0.7212 |
| mAP50-95 | 0.3887 |

### Class-wise performance
| Defect | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| crazing | 0.4800 | 0.3480 | 0.4383 | 0.1842 |
| inclusion | 0.6690 | 0.7860 | 0.8057 | 0.4666 |
| patches | 0.8040 | 0.8740 | 0.9102 | 0.5834 |
| pitted_surface | 0.7250 | 0.7560 | 0.8328 | 0.4574 |
| rolled-in_scale | 0.4530 | 0.6080 | 0.5751 | 0.2516 |
| scratches | 0.4230 | 0.8760 | 0.7651 | 0.3889 |

## Project workflow
1. Convert Pascal VOC XML bounding boxes into YOLO format.
2. Train YOLO11n on the training partition.
3. Validate using the internal validation partition.
4. Evaluate on the untouched final test partition.
5. Generate defect predictions and evaluation charts.

## Reproduce
Install dependencies using `pip install -r requirements.txt`.

Prepare data:
`python src/prepare_dataset.py --source-root /path/to/NEU-DET`

Train:
`python src/train.py --data p3_yolo_dataset/data.yaml`

Evaluate:
`python src/evaluate.py --data p3_yolo_dataset/data.yaml --weights runs/yolo11n_neu_defect/weights/best.pt`

Predict:
`python src/predict.py --weights runs/yolo11n_neu_defect/weights/best.pt --source /path/to/images`

## Limitations
- Crazing and rolled-in scale are harder to detect.
- Some predictions overlap or duplicate.
- The dataset is relatively small and class-imbalanced.
- This is an educational prototype, not a validated production system.
- No factory camera, conveyor, PLC or automatic rejection hardware is integrated.

## Model weights
The trained best.pt checkpoint is approximately 5.21 MB.
It is not included in this package. Recreate it by training the model.

## Repository structure
- src/: dataset preparation, training, evaluation and prediction scripts
- docs/: report and experiment notes
- dataset/: dataset download and licence information
- models/: checkpoint information
- results/: metrics, training history and charts

# Project Report
## Industrial Surface Defect Detection Using YOLO11n

### Abstract
A custom-trained YOLO11n detector was developed for six steel surface
defect categories. XML annotations were converted to YOLO format.
The final model was evaluated on 360 held-out images.

### Dataset
- Images: 1,800
- Annotated objects: 4,189
- Training: 1,152
- Internal validation: 288
- Final test: 360

### Training
- Model: pretrained YOLO11n
- Epochs: 30
- Input size: 320 x 320
- Batch: 16
- Optimizer: AdamW, automatically selected
- Hardware: Tesla T4

### Final test results
- Precision: 0.5924
- Recall: 0.7081
- mAP50: 0.7212
- mAP50-95: 0.3887

### Class-wise results
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| crazing | 0.4800 | 0.3480 | 0.4383 | 0.1842 |
| inclusion | 0.6690 | 0.7860 | 0.8057 | 0.4666 |
| patches | 0.8040 | 0.8740 | 0.9102 | 0.5834 |
| pitted_surface | 0.7250 | 0.7560 | 0.8328 | 0.4574 |
| rolled-in_scale | 0.4530 | 0.6080 | 0.5751 | 0.2516 |
| scratches | 0.4230 | 0.8760 | 0.7651 | 0.3889 |

### Discussion
Patches achieved the strongest test mAP50. Crazing was the most
challenging category. Scratches showed high recall but lower precision,
indicating additional false detections. The overall precision indicates
that false positives remain an important limitation.

### Limitations and future scope
The dataset is small and imbalanced. Future work should add more
difficult examples, improve defect-specific augmentation, tune
confidence and NMS thresholds using validation data, and evaluate
on independently sourced industrial images.

### Conclusion
The detector achieved an independent test mAP50 of 0.7212 and mAP50-95 of 0.3887.
It demonstrates a working educational steel defect detection prototype.

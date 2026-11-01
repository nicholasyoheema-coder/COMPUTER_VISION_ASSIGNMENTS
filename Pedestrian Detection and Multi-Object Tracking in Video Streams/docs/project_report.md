# Project Report
## Multi-Object Pedestrian Tracking Using YOLO11n and ByteTrack

### 1. Abstract

This project implements multi-object pedestrian tracking in video
using pretrained YOLO11n detection and ByteTrack identity association.
The MOT17-02-FRCNN sequence was processed over 600 frames.
Tracking was evaluated against active pedestrian ground-truth annotations.

### 2. Dataset

- Sequence: MOT17-02-FRCNN
- Frames: 600
- Resolution: 1920 × 1080
- Frame rate: 30 FPS
- Active pedestrian annotations: 18,581
- Ground-truth pedestrian identities: 62

Other MOT17 categories and inactive/ignored annotations were excluded.

### 3. Methodology

YOLO11n pretrained COCO weights detect people using class 0.
ByteTrack associates detections across successive frames and assigns
tracking IDs. Two configurations were tested.

Baseline: confidence 0.25, image size 640, default ByteTrack settings.
Tuned: confidence 0.10, image size 960, track buffer 60,
high threshold 0.25, low threshold 0.10, matching threshold 0.80.

Evaluation used an IoU threshold of 0.50 and MOTMetrics.

### 4. Results

| Metric | Baseline | Tuned |
|---|---:|---:|
| MOTA | 0.1851 | 0.2403 |
| IDF1 | 0.2718 | 0.3594 |
| Precision | 0.8458 | 0.7750 |
| Recall | 0.2303 | 0.3447 |
| ID switches | 61 | 80 |
| Fragmentations | 158 | 137 |
| False positives | 780 | 1859 |
| Misses | 14301 | 12177 |

### 5. Discussion

The tuned configuration increased MOTA from 0.1851 to 0.2403
and IDF1 from 0.2718 to 0.3594. Recall increased from 0.2303
to 0.3447, while precision decreased from 0.8458 to 0.7750.
ID switches increased from 61 to 80, while fragmentations decreased
from 158 to 137. The tuned approach therefore detects more pedestrians
but incurs more false positives and identity switches.

### 6. Limitations

The experiment evaluates only one MOT17 sequence. Tracking identity
consistency remains imperfect, and the detector misses many annotated
pedestrians. The results should not be presented as full MOT17 benchmark
performance.

### 7. Conclusion

YOLO11n and ByteTrack were integrated into a complete video tracking
pipeline. On MOT17-02-FRCNN, the tuned configuration achieved MOTA
0.2403 and IDF1 0.3594. The comparison shows a recall improvement
with trade-offs in precision and ID switches.

# Multi-Object Pedestrian Tracking Using YOLO11n and ByteTrack

A video-based multi-object tracking project using pretrained YOLO11n
for pedestrian detection and ByteTrack for identity association.

## Dataset

**MOT17-02-FRCNN** from the MOT17 multi-object tracking benchmark.

| Property | Value |
|---|---:|
| Frames | 600 |
| Resolution | 1920 × 1080 |
| Frame rate | 30 FPS |
| Active pedestrian annotations | 18,581 |
| Ground-truth pedestrian identities | 62 |

Only active pedestrian annotations (class 1, active flag 1) were used
for pedestrian tracking evaluation.

## Method

1. Detect people using pretrained YOLO11n (COCO person class 0).
2. Process frames sequentially.
3. Associate detections across frames using ByteTrack.
4. Preserve track IDs and export frame-wise bounding boxes.
5. Evaluate predicted identities against MOT ground truth.

## Final selected configuration

- Detector: YOLO11n pretrained COCO
- Tracker: ByteTrack
- Confidence: 0.10
- Image size: 960
- Track buffer: 60
- IoU evaluation threshold: 0.50

## Results on MOT17-02-FRCNN

| Metric | Baseline | Tuned |
|---|---:|---:|
| MOTA | 0.1851 | **0.2403** |
| IDF1 | 0.2718 | **0.3594** |
| Precision | 0.8458 | 0.7750 |
| Recall | 0.2303 | **0.3447** |
| ID switches | 61 | 80 |
| Fragmentations | 158 | **137** |
| False positives | 780 | 1859 |
| Misses | 14301 | **12177** |

The tuned configuration achieved MOTA **0.2403**
and IDF1 **0.3594** on this sequence.

It improves MOTA, IDF1 and recall compared with the baseline,
but it also increases false positives and ID switches.
This is a single-sequence experiment, not a full MOT17 benchmark score.

## Installation

`pip install -r requirements.txt`

## Reproduce tracking

Download MOT17-02-FRCNN separately and extract it locally.

```bash
python src/track.py \
  --sequence-dir /path/to/MOT17-02-FRCNN \
  --output-dir tracking_output \
  --conf 0.10 --imgsz 960 --track-buffer 60
```

## Evaluate tracking

```bash
python src/evaluate.py \
  --gt /path/to/MOT17-02-FRCNN/gt/gt.txt \
  --tracks tracking_output/tracks.csv \
  --frames 600 \
  --output-dir tracking_metrics
```

## Repository structure

- `src/track.py`: YOLO11n + ByteTrack tracking pipeline
- `src/evaluate.py`: MOTMetrics evaluation
- `configs/bytetrack_tuned.yaml`: tuned ByteTrack settings
- `docs/`: report and experiment notes
- `dataset/`: dataset access and annotation notes
- `results/metrics/`: metrics and baseline comparison
- `results/figures/`: aggregate comparison charts

## Limitations

- Evaluation covers one MOT17 sequence only.
- The tuned tracker generates more false positives and ID switches.
- IDF1 remains limited, indicating identity association challenges.
- No appearance embedding or re-identification model was added.
- No full MOT17 benchmark-wide result is claimed.

The raw dataset, videos, frame images and frame-level trajectory CSVs
are not redistributed in this repository. Obtain the benchmark from
its official source and follow its usage terms.

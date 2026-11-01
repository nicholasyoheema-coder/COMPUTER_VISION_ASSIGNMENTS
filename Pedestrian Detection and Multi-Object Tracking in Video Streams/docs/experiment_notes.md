# Experiment Notes

Sequence: MOT17-02-FRCNN
Frames: 600
Resolution: 1920x1080
Frame rate: 30
Ground-truth active pedestrian rows: 18,581
Ground-truth pedestrian identities: 62

Baseline: YOLO11n, confidence 0.25, image size 640, default ByteTrack.
Tuned: YOLO11n, confidence 0.10, image size 960, track buffer 60.

Baseline MOTA: 0.1851
Tuned MOTA: 0.2403
Baseline IDF1: 0.2718
Tuned IDF1: 0.3594
Baseline recall: 0.2303
Tuned recall: 0.3447

Tuned configuration improves MOTA, IDF1, recall and fragmentation
count but lowers precision and increases ID switches.
Only one MOT17 sequence was evaluated.

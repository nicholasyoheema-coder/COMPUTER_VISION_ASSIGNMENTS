# Dataset Information

Dataset: ROD - Obstacle Detection Dataset

Kaggle source:
https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset

The dataset contains 24,326 images and 25 object classes.

| Split | Images |
|---|---:|
| Train | 19,186 |
| Validation | 3,511 |
| Test | 1,629 |
| Total | 24,326 |

The annotations contained both YOLO bounding boxes and YOLO segmentation polygons.

For this object-detection project, polygon outlines were converted into axis-aligned bounding boxes using their minimum and maximum normalized coordinates.

A maximum normalized boundary overflow of 0.000010416667 was measured in 190 source boxes. These were clipped to the image boundaries. 99 empty annotation files were preserved as background images.

The final detection dataset contained 40,195 valid boxes and 25 classes.

The raw dataset is not included in this repository. Download it from Kaggle and comply with its current terms and licensing conditions. The prepared dataset can be regenerated using `src/prepare_dataset.py`.

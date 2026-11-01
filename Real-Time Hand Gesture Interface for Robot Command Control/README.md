# Gesture Recognition for Robot Control

A real-time, vision-based gesture recognition prototype that maps hand gestures to simulated robot movement commands.

## Features
- Hand detection and gesture classification using MediaPipe.
- Five gesture-to-command mappings.
- Webcam-based live demonstration.
- Five-frame voting to reduce unstable commands.
- STOP for missing hands, unknown gestures and low-confidence predictions.
- Emergency stop behaviour.

## Gesture-to-command mapping

| Gesture | Robot command |
|---|---|
| Open Palm | STOP |
| Closed Fist | MOVE BACKWARD |
| Pointing Up | MOVE FORWARD |
| Thumb Down | TURN LEFT |
| Thumb Up | TURN RIGHT |

## System architecture

Webcam -> MediaPipe hand detection -> Hand landmarks -> Gesture classification -> Confidence check -> Command smoothing -> Robot command

## Technology
- Python
- MediaPipe Gesture Recognizer
- OpenCV
- NumPy
- Google Colab for initial testing

## Run locally on macOS

Use Python 3.10–3.12.

1. Open Terminal in this project folder.
2. Create and activate a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

4. Run the live webcam demo:

   python src/webcam_robot_control.py

Allow camera access to the terminal application when macOS requests it.
Press Q to quit.

## Test results

Five manually supplied test images were evaluated:
- Open Palm: 0.6034
- Closed Fist: 0.9108
- Pointing Up: 0.5324
- Thumb Down: 0.9197
- Thumb Up: 0.6793

All five sample images were classified as expected (5/5).

These are illustrative sample tests, not a representative accuracy benchmark. A larger independent test set is required to measure generalisation.

## Model and training note

This project uses Google's pretrained MediaPipe Gesture Recognizer model. We did not train a custom MobileNetV2 model or fine-tune the pretrained model. The 4.82 GB HandGesture-5G dataset was not used.

## Safety and limitations
- This is a software-only command simulation, not a physical robot controller.
- The 0.50 confidence threshold is provisional and needs broader validation.
- Only one hand is configured for recognition.
- The built-in classifier recognises a fixed set of gestures; it does not recognise arbitrary hand signs.
- The system is not intended for safety-critical or human-life-critical control.
- Do not connect these commands directly to motors without independent hardware safety interlocks.

## Project structure

- `src/robot_controller.py` - gesture-to-command mapping and safety logic
- `src/webcam_robot_control.py` - real-time webcam application
- `models/gesture_recognizer.task` - pretrained MediaPipe model
- `results/metrics/gesture_test_results.csv` - five-image sample results
- `docs/` - implementation notes and references

## References
- MediaPipe Gesture Recognizer: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer
- MediaPipe Python guide: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python
- Official model card: https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf

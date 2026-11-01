# Technical Report: Gesture Recognition for Robot Control

## 1. Objective
Develop a vision-based interface that recognises static hand gestures and converts them into simulated robot movement commands.

## 2. Methodology
The system uses Google's pretrained MediaPipe Gesture Recognizer.

Processing pipeline:
1. Capture an image or webcam frame.
2. Detect the hand.
3. Extract 21 hand landmarks.
4. Classify the hand gesture.
5. Check prediction confidence.
6. Apply temporal command smoothing.
7. Generate a simulated robot command.

The pretrained recognizer uses a hand-landmark and gesture-classification pipeline. This project does not train a new neural network.

## 3. Hardware and software
- Computer: Laptop with webcam
- Language: Python
- Libraries: MediaPipe, OpenCV, NumPy
- Development environment: Google Colab
- Final demonstration environment: macOS

## 4. Command mapping

| Gesture | Command |
|---|---|
| Open Palm | STOP |
| Closed Fist | MOVE BACKWARD |
| Pointing Up | MOVE FORWARD |
| Thumb Down | TURN LEFT |
| Thumb Up | TURN RIGHT |

## 5. Safety logic
- Confidence below 0.50 results in STOP.
- Unknown gestures result in STOP.
- Missing hand detection results in STOP.
- Open Palm immediately generates STOP.
- Movement commands require at least three matching predictions in a five-frame history.
- Emergency stop clears the command history.

The 0.50 threshold is provisional and requires further testing.

## 6. Sample testing

| Test image | Expected gesture | Recognised gesture | Confidence | Correct |
|---|---|---|---:|---|
| images.jpeg | Open Palm | Open Palm | 0.6034 | Yes |
| images-2.jpeg | Closed Fist | Closed Fist | 0.9108 | Yes |
| images-3.jpeg | Pointing Up | Pointing Up | 0.5324 | Yes |
| images-4.jpeg | Thumb Down | Thumb Down | 0.9197 | Yes |
| images-5.jpeg | Thumb Up | Thumb Up | 0.6793 | Yes |

Five out of five supplied sample images were classified as expected. This is a small functional test, not a statistically representative model accuracy result.

## 7. Limitations
- Uses pretrained gesture categories rather than a custom-trained classifier.
- No custom dataset was used.
- Only one hand is configured.
- Only static gestures supported by the pretrained classifier are mapped.
- Performance under different lighting, distances, backgrounds and users needs broader evaluation.
- No physical robot or motor driver is connected.
- The live webcam application has been created but must be tested locally on the target Mac.

## 8. Conclusion
The prototype demonstrates a complete gesture-to-command software pipeline with confidence filtering, temporal smoothing and STOP behaviour for uncertain inputs. It provides a foundation for future integration with a robotic platform, subject to additional validation and independent hardware safety controls.

## 9. References
1. Google AI Edge, MediaPipe Gesture Recognizer: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer
2. MediaPipe Gesture Recognizer Python guide: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python
3. Google, Hand Gesture Classification Model Card: https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf

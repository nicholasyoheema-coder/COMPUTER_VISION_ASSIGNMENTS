
import cv2
import mediapipe as mp
import time
from pathlib import Path

from robot_controller import GestureCommandController

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "gesture_recognizer.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

controller = GestureCommandController(
    threshold=0.50,
    window=5,
    votes_required=3
)

recognizer = GestureRecognizer.create_from_options(options)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    recognizer.close()
    raise RuntimeError(
        "Webcam unavailable. Check camera permissions and connection."
    )

print("Gesture robot-control demo started.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame unavailable.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000

        result = recognizer.recognize_for_video(
            mp_image,
            timestamp_ms
        )

        gesture_name = "No hand"
        confidence = 0.0

        if result.gestures and result.gestures[0]:
            prediction = result.gestures[0][0]
            gesture_name = prediction.category_name
            confidence = prediction.score

        command, status = controller.update(
            gesture_name,
            confidence
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"ROBOT: {command}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0) if command != "STOP" else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow("Gesture-Based Robot Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    controller.emergency_stop()
    camera.release()
    cv2.destroyAllWindows()
    recognizer.close()
    print("Demo stopped. Emergency STOP issued.")

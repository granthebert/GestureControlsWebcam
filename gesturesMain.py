import cv2
import pyautogui
import mediapipe as mp
import os
import time
import urllib.request

# Webcam
cap = cv2.VideoCapture(0)

# Gesture tuning
FIST_MOVE_STEP_PX = 15
PINCH_MOVE_STEP_PX = 15
FIST_MIN_CONFIDENCE_FRAMES = 3
VOLUME_PRESS_INTERVAL = 0.02
GESTURE_CONFIRM_FRAMES = 6
GESTURE_COOLDOWN_SECONDS = 1.0
DISCORD_LEAVE_HOTKEY = ("ctrl", "alt", "shift", "d")


def distance(first, second):
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


def finger_is_extended(hand_landmarks, tip_id, pip_id):
    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y


def finger_is_folded(hand_landmarks, tip_id, pip_id):
    return hand_landmarks[tip_id].y > hand_landmarks[pip_id].y


def palm_width(hand_landmarks):
    return distance(hand_landmarks[5], hand_landmarks[17])


def is_fist(hand_landmarks):
    """Return True when the four main fingers look folded into the palm."""
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    folded_fingers = 0

    for tip_id, pip_id in finger_pairs:
        if hand_landmarks[tip_id].y > hand_landmarks[pip_id].y:
            folded_fingers += 1

    return folded_fingers >= 4


def is_open_palm(hand_landmarks):
    fingers_open = all(
        finger_is_extended(hand_landmarks, tip_id, pip_id)
        for tip_id, pip_id in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    thumb_open = distance(hand_landmarks[4], hand_landmarks[9]) > palm_width(hand_landmarks) * 0.75
    return fingers_open and thumb_open


def is_pointing(hand_landmarks):
    return (
        finger_is_extended(hand_landmarks, 8, 6)
        and finger_is_folded(hand_landmarks, 12, 10)
        and finger_is_folded(hand_landmarks, 16, 14)
        and finger_is_folded(hand_landmarks, 20, 18)
    )


def is_peace_sign(hand_landmarks):
    return (
        finger_is_extended(hand_landmarks, 8, 6)
        and finger_is_extended(hand_landmarks, 12, 10)
        and finger_is_folded(hand_landmarks, 16, 14)
        and finger_is_folded(hand_landmarks, 20, 18)
    )


def is_thumbs_down(hand_landmarks):
    fingers_folded = all(
        finger_is_folded(hand_landmarks, tip_id, pip_id)
        for tip_id, pip_id in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    thumb_below_fingers = hand_landmarks[4].y > hand_landmarks[8].y
    thumb_extended = distance(hand_landmarks[4], hand_landmarks[2]) > palm_width(hand_landmarks) * 0.55
    return fingers_folded and thumb_below_fingers and thumb_extended


def is_pinching(hand_landmarks):
    return distance(hand_landmarks[4], hand_landmarks[8]) < palm_width(hand_landmarks) * 0.35


def hand_center(hand_landmarks, frame_width, frame_height):
    palm_landmarks = (0, 5, 9, 13, 17)
    x = sum(hand_landmarks[i].x for i in palm_landmarks) / len(palm_landmarks)
    y = sum(hand_landmarks[i].y for i in palm_landmarks) / len(palm_landmarks)
    return int(x * frame_width), int(y * frame_height)


def adjust_volume_from_fist(current_y, state):
    movement = current_y - state["anchor_y"]
    current_step = int(movement / FIST_MOVE_STEP_PX)
    step_delta = current_step - state["last_step"]

    if step_delta == 0:
        return

    key = "volumedown" if step_delta > 0 else "volumeup"
    pyautogui.press(key, presses=abs(step_delta), interval=VOLUME_PRESS_INTERVAL)
    state["last_step"] = current_step


def adjust_player_volume_from_pinch(current_y, state):
    movement = current_y - state["anchor_y"]
    current_step = int(movement / PINCH_MOVE_STEP_PX)
    step_delta = current_step - state["last_step"]

    if step_delta == 0:
        return

    # Moving down lowers browser/player volume; moving up raises it.
    key = "down" if step_delta > 0 else "up"
    pyautogui.press(key, presses=abs(step_delta), interval=VOLUME_PRESS_INTERVAL)
    state["last_step"] = current_step


def recognize_gesture(hand_landmarks):
    if is_open_palm(hand_landmarks):
        return "open_palm"
    if is_pinching(hand_landmarks):
        return "pinch"
    if is_thumbs_down(hand_landmarks):
        return "thumbs_down"
    if is_peace_sign(hand_landmarks):
        return "peace"
    if is_pointing(hand_landmarks):
        return "pointing"
    if is_fist(hand_landmarks):
        return "fist"
    return None


def handle_confirmed_gesture(gesture_name, action, state, now):
    if gesture_name != state["last_seen"]:
        state["last_seen"] = gesture_name
        state["frames"] = 0
        state["triggered"] = False

    state["frames"] += 1
    if state["frames"] < GESTURE_CONFIRM_FRAMES:
        return

    if state["triggered"]:
        return

    if now - state["last_action_time"] < GESTURE_COOLDOWN_SECONDS:
        return

    action()
    state["last_action_time"] = now
    state["triggered"] = True


fist_state = {
    "active": False,
    "confidence_frames": 0,
    "anchor_y": None,
    "last_step": 0,
}
pinch_state = {
    "active": False,
    "anchor_y": None,
    "last_step": 0,
}
one_shot_state = {
    "last_seen": None,
    "frames": 0,
    "triggered": False,
    "last_action_time": 0,
}
open_palm_state = {
    "last_seen": None,
    "frames": 0,
    "triggered": False,
    "last_action_time": 0,
}
gestures_paused = False

# Choose API depending on installed MediaPipe
USE_SOLUTIONS = hasattr(mp, "solutions")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

if USE_SOLUTIONS:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_draw = mp.solutions.drawing_utils
    API_MODE = "solutions"
else:
    # Use the newer tasks API. Download the .task model if missing.
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as e:
        raise ImportError(
            "Installed MediaPipe is missing the tasks API required for this fallback: "
            f"{e}"
        )

    if not os.path.exists(MODEL_PATH):
        try:
            print(f"Downloading model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                "Failed to download hand_landmarker.task model.\n"
                "You can download it manually from:\n"
                f"{MODEL_URL}\n"
                f"Error: {e}"
            )

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    API_MODE = "tasks"

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame.shape
    detected_gesture = None
    center = None

    if API_MODE == "solutions":
        # legacy MediaPipe solutions API
        result = hands.process(rgb_frame)
        if result and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                detected_gesture = recognize_gesture(landmarks)
                center = hand_center(landmarks, w, h)
                break
    else:
        # mediapipe.tasks API
        try:
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
            detection_result = detector.detect(mp_image)
        except Exception:
            detection_result = None

        if detection_result and getattr(detection_result, "hand_landmarks", None):
            for hand_landmarks in detection_result.hand_landmarks:
                # hand_landmarks is a list of landmarks with normalized x,y
                detected_gesture = recognize_gesture(hand_landmarks)
                center = hand_center(hand_landmarks, w, h)
                break

    now = time.monotonic()

    if detected_gesture == "open_palm":
        handle_confirmed_gesture(
            "open_palm",
            lambda: globals().__setitem__("gestures_paused", not gestures_paused),
            open_palm_state,
            now,
        )
    else:
        open_palm_state["last_seen"] = None
        open_palm_state["frames"] = 0
        open_palm_state["triggered"] = False

    if gestures_paused:
        fist_state["active"] = False
        fist_state["confidence_frames"] = 0
        pinch_state["active"] = False
        cv2.putText(
            frame,
            "Gestures paused",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
    elif detected_gesture == "fist" and center:
        fist_state["confidence_frames"] += 1
        cx, cy = center
        pinch_state["active"] = False

        if not fist_state["active"] and fist_state["confidence_frames"] >= FIST_MIN_CONFIDENCE_FRAMES:
            fist_state["active"] = True
            fist_state["anchor_y"] = cy
            fist_state["last_step"] = 0

        if fist_state["active"]:
            adjust_volume_from_fist(cy, fist_state)
            cv2.circle(frame, (cx, cy), 14, (0, 255, 0), -1)
            cv2.line(frame, (0, fist_state["anchor_y"]), (w, fist_state["anchor_y"]), (0, 255, 255), 2)
            cv2.putText(
                frame,
                "Fist volume mode",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
    elif detected_gesture == "pinch" and center:
        fist_state["active"] = False
        fist_state["confidence_frames"] = 0
        cx, cy = center

        if not pinch_state["active"]:
            pinch_state["active"] = True
            pinch_state["anchor_y"] = cy
            pinch_state["last_step"] = 0

        adjust_player_volume_from_pinch(cy, pinch_state)
        cv2.circle(frame, (cx, cy), 14, (255, 0, 255), -1)
        cv2.line(frame, (0, pinch_state["anchor_y"]), (w, pinch_state["anchor_y"]), (255, 0, 255), 2)
        cv2.putText(
            frame,
            "Pinch arrow volume",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )
    elif detected_gesture == "pointing":
        fist_state["active"] = False
        fist_state["confidence_frames"] = 0
        pinch_state["active"] = False
        handle_confirmed_gesture("pointing", lambda: pyautogui.press("playpause"), one_shot_state, now)
        cv2.putText(
            frame,
            "Media pause/play",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
                2,
            )
    elif detected_gesture == "thumbs_down":
        fist_state["active"] = False
        fist_state["confidence_frames"] = 0
        pinch_state["active"] = False
        handle_confirmed_gesture("thumbs_down", lambda: pyautogui.hotkey("alt", "f4"), one_shot_state, now)
        cv2.putText(
            frame,
            "Alt+F4",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 255),
            2,
        )
    elif detected_gesture == "peace":
        fist_state["active"] = False
        fist_state["confidence_frames"] = 0
        pinch_state["active"] = False
        handle_confirmed_gesture(
            "peace",
            lambda: pyautogui.hotkey(*DISCORD_LEAVE_HOTKEY),
            one_shot_state,
            now,
        )
        cv2.putText(
            frame,
            "Discord hotkey",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            2,
        )
    else:
        fist_state["active"] = False
        fist_state["confidence_frames"] = 0
        fist_state["anchor_y"] = None
        fist_state["last_step"] = 0
        pinch_state["active"] = False
        pinch_state["anchor_y"] = None
        pinch_state["last_step"] = 0
        if not detected_gesture:
            one_shot_state["last_seen"] = None
            one_shot_state["frames"] = 0
            one_shot_state["triggered"] = False

    if detected_gesture and center:
        cv2.putText(
            frame,
            f"Gesture: {detected_gesture}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Gesture Control", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

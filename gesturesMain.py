import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import pyautogui
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from testGUI import GestureAction, GestureDashboard


FIST_MOVE_STEP_PX = 15
PINCH_MOVE_STEP_PX = 15
FIST_MIN_CONFIDENCE_FRAMES = 3
VOLUME_PRESS_INTERVAL = 0.02
GESTURE_CONFIRM_FRAMES = 6
GESTURE_COOLDOWN_SECONDS = 1.0
COMMAND_SWITCH_DELAY_SECONDS = 0.6
DISCORD_LEAVE_HOTKEY = ("ctrl", "alt", "shift", "d")

SWIPE_MIN_DISTANCE_PX = 110
SWIPE_DIRECTION_RATIO = 1.25
SWIPE_COOLDOWN_SECONDS = 0.55

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def distance(first, second):
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


def finger_is_extended(hand_landmarks, tip_id, pip_id):
    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y


def finger_is_folded(hand_landmarks, tip_id, pip_id):
    return hand_landmarks[tip_id].y > hand_landmarks[pip_id].y


def palm_width(hand_landmarks):
    return distance(hand_landmarks[5], hand_landmarks[17])


def is_fist(hand_landmarks):
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


def normalized_hand_center(hand_landmarks):
    palm_landmarks = (0, 5, 9, 13, 17)
    x = sum(hand_landmarks[i].x for i in palm_landmarks) / len(palm_landmarks)
    y = sum(hand_landmarks[i].y for i in palm_landmarks) / len(palm_landmarks)
    return x, y


def is_hand_circle(first_hand, second_hand):
    first_width = palm_width(first_hand)
    second_width = palm_width(second_hand)
    average_width = (first_width + second_width) / 2
    if average_width <= 0:
        return False

    first_center = normalized_hand_center(first_hand)
    second_center = normalized_hand_center(second_hand)
    palm_gap = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
    if palm_gap < average_width * 1.15:
        return False

    thumb_gap = distance(first_hand[4], second_hand[4])
    if thumb_gap > average_width * 0.52:
        return False

    fingertip_pairs = ((8, 8), (12, 12), (16, 16), (20, 20))
    touching_fingers = 0
    for first_tip, second_tip in fingertip_pairs:
        if distance(first_hand[first_tip], second_hand[second_tip]) < average_width * 0.58:
            touching_fingers += 1

    finger_cluster_center_x = (
        first_hand[8].x + first_hand[12].x + first_hand[16].x + first_hand[20].x
        + second_hand[8].x + second_hand[12].x + second_hand[16].x + second_hand[20].x
    ) / 8
    finger_cluster_center_y = (
        first_hand[8].y + first_hand[12].y + first_hand[16].y + first_hand[20].y
        + second_hand[8].y + second_hand[12].y + second_hand[16].y + second_hand[20].y
    ) / 8
    thumb_center_x = (first_hand[4].x + second_hand[4].x) / 2
    thumb_center_y = (first_hand[4].y + second_hand[4].y) / 2
    cluster_gap = (
        (finger_cluster_center_x - thumb_center_x) ** 2
        + (finger_cluster_center_y - thumb_center_y) ** 2
    ) ** 0.5

    return touching_fingers >= 3 and cluster_gap > average_width * 0.55


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


def make_one_shot_state():
    return {
        "last_seen": None,
        "frames": 0,
        "triggered": False,
        "last_action_time": 0,
    }


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


class SwipeTracker:
    def __init__(self):
        self.anchor = None
        self.last_swipe_time = 0

    def reset(self):
        self.anchor = None

    def update(self, center, now):
        if center is None:
            self.anchor = None
            return None

        if self.anchor is None:
            self.anchor = center
            return None

        if now - self.last_swipe_time < SWIPE_COOLDOWN_SECONDS:
            self.anchor = center
            return None

        dx = center[0] - self.anchor[0]
        dy = center[1] - self.anchor[1]
        action = None

        if abs(dx) >= SWIPE_MIN_DISTANCE_PX and abs(dx) > abs(dy) * SWIPE_DIRECTION_RATIO:
            action = GestureAction.SWIPE_RIGHT if dx > 0 else GestureAction.SWIPE_LEFT
        elif abs(dy) >= SWIPE_MIN_DISTANCE_PX and abs(dy) > abs(dx) * SWIPE_DIRECTION_RATIO:
            action = GestureAction.SWIPE_DOWN if dy > 0 else GestureAction.SWIPE_UP

        if action is not None:
            self.anchor = center
            self.last_swipe_time = now
            return action

        if abs(dx) < 30 and abs(dy) < 30:
            self.anchor = center

        return None


class GestureController:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.cap = cv2.VideoCapture(0)

        self.fist_state = {
            "active": False,
            "confidence_frames": 0,
            "anchor_y": None,
            "last_step": 0,
        }
        self.pinch_state = {
            "active": False,
            "anchor_y": None,
            "last_step": 0,
        }
        self.one_shot_state = make_one_shot_state()
        self.open_palm_state = make_one_shot_state()
        self.gesture_candidate = None
        self.gesture_candidate_since = 0
        self.desktop_gestures_paused = False
        self.gui_control_active = True
        self.swipe_tracker = SwipeTracker()

        self._setup_mediapipe()
        self.dashboard.set_control_surface_active(self.gui_control_active)

    def _setup_mediapipe(self):
        self.use_solutions = hasattr(mp, "solutions")
        if self.use_solutions:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.api_mode = "solutions"
            return

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:
            raise ImportError(
                "Installed MediaPipe is missing the tasks API required for this fallback: "
                f"{exc}"
            )

        if not os.path.exists(MODEL_PATH):
            try:
                print(f"Downloading model to {MODEL_PATH}...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to download hand_landmarker.task model.\n"
                    "You can download it manually from:\n"
                    f"{MODEL_URL}\n"
                    f"Error: {exc}"
                )

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.api_mode = "tasks"

    def process_frame(self):
        success, frame = self.cap.read()
        if not success:
            QApplication.quit()
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        detected_gesture, center = self._detect_hand(rgb_frame, frame, width, height)
        now = time.monotonic()
        detected_gesture = self._stabilize_command_switch(detected_gesture, now, frame)

        if detected_gesture == "hand_circle":
            handle_confirmed_gesture(
                detected_gesture,
                self._toggle_gui_control,
                self.open_palm_state,
                now,
            )
        else:
            self._reset_open_palm()

        if self.gui_control_active:
            self._handle_gui_control(detected_gesture, center, now, frame)
        else:
            self._handle_desktop_control(detected_gesture, center, now, frame, width)

        self._draw_status(frame, detected_gesture, center, height)
        cv2.imshow("Gesture Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            QApplication.quit()
        elif key == ord("m"):
            self._toggle_gui_control()
        elif key == ord("p"):
            self.desktop_gestures_paused = not self.desktop_gestures_paused

    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if getattr(self, "api_mode", None) == "solutions":
            self.hands.close()

    def _detect_hand(self, rgb_frame, draw_frame, width, height):
        detected_gesture = None
        center = None

        if self.api_mode == "solutions":
            result = self.hands.process(rgb_frame)
            if result and result.multi_hand_landmarks:
                all_landmarks = [
                    hand_landmarks.landmark
                    for hand_landmarks in result.multi_hand_landmarks
                ]
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        draw_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )
                if len(all_landmarks) >= 2 and is_hand_circle(all_landmarks[0], all_landmarks[1]):
                    detected_gesture = "hand_circle"
                    first_center = hand_center(all_landmarks[0], width, height)
                    second_center = hand_center(all_landmarks[1], width, height)
                    center = (
                        (first_center[0] + second_center[0]) // 2,
                        (first_center[1] + second_center[1]) // 2,
                    )
                else:
                    landmarks = all_landmarks[0]
                    detected_gesture = recognize_gesture(landmarks)
                    center = hand_center(landmarks, width, height)
        else:
            try:
                mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
                detection_result = self.detector.detect(mp_image)
            except Exception:
                detection_result = None

            if detection_result and getattr(detection_result, "hand_landmarks", None):
                all_landmarks = detection_result.hand_landmarks
                if len(all_landmarks) >= 2 and is_hand_circle(all_landmarks[0], all_landmarks[1]):
                    detected_gesture = "hand_circle"
                    first_center = hand_center(all_landmarks[0], width, height)
                    second_center = hand_center(all_landmarks[1], width, height)
                    center = (
                        (first_center[0] + second_center[0]) // 2,
                        (first_center[1] + second_center[1]) // 2,
                    )
                else:
                    hand_landmarks = all_landmarks[0]
                    detected_gesture = recognize_gesture(hand_landmarks)
                    center = hand_center(hand_landmarks, width, height)

        return detected_gesture, center

    def _stabilize_command_switch(self, detected_gesture, now, frame):
        raw_detected_gesture = detected_gesture
        if raw_detected_gesture != self.gesture_candidate:
            self.gesture_candidate = raw_detected_gesture
            self.gesture_candidate_since = now
            self._reset_motion_states()
            self._reset_one_shot()
            self._reset_open_palm()

        if self.gui_control_active and raw_detected_gesture == "open_palm":
            return raw_detected_gesture

        if raw_detected_gesture and now - self.gesture_candidate_since < COMMAND_SWITCH_DELAY_SECONDS:
            cv2.putText(
                frame,
                "Hold gesture...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (180, 180, 180),
                2,
            )
            return None

        return detected_gesture

    def _handle_gui_control(self, detected_gesture, center, now, frame):
        self._reset_motion_states()
        swipe_action = self.swipe_tracker.update(
            center if detected_gesture == "open_palm" else None,
            now,
        )
        if swipe_action is not None:
            self.dashboard.dispatch_gesture(swipe_action)

        if detected_gesture == "pointing":
            handle_confirmed_gesture(
                "gui_select",
                lambda: self.dashboard.dispatch_gesture(GestureAction.SELECT),
                self.one_shot_state,
                now,
            )
        elif detected_gesture == "peace":
            handle_confirmed_gesture(
                "gui_lock",
                self._toggle_dashboard_lock,
                self.one_shot_state,
                now,
            )
        elif detected_gesture == "thumbs_down":
            handle_confirmed_gesture(
                "gui_back",
                self._dashboard_back_or_exit,
                self.one_shot_state,
                now,
            )
        elif detected_gesture not in ("open_palm", "hand_circle"):
            self._reset_one_shot()

        cv2.putText(
            frame,
            "GUI mode: open-palm swipe | peace lock | point select | circle exit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (78, 220, 255),
            2,
        )

    def _handle_desktop_control(self, detected_gesture, center, now, frame, width):
        self.swipe_tracker.reset()
        if self.desktop_gestures_paused:
            self._reset_motion_states()
            cv2.putText(
                frame,
                "Desktop gestures paused (press p)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            return

        if detected_gesture == "fist" and center:
            self._handle_fist_volume(center, frame, width)
        elif detected_gesture == "pinch" and center:
            self._handle_pinch_volume(center, frame, width)
        elif detected_gesture == "pointing":
            self._reset_motion_states()
            handle_confirmed_gesture(
                "pointing",
                lambda: pyautogui.press("playpause"),
                self.one_shot_state,
                now,
            )
            self._draw_mode_text(frame, "Media pause/play", (255, 255, 0))
        elif detected_gesture == "thumbs_down":
            self._reset_motion_states()
            handle_confirmed_gesture(
                "thumbs_down",
                lambda: pyautogui.hotkey("alt", "f4"),
                self.one_shot_state,
                now,
            )
            self._draw_mode_text(frame, "Alt+F4", (0, 200, 255))
        elif detected_gesture == "peace":
            self._reset_motion_states()
            handle_confirmed_gesture(
                "peace",
                lambda: pyautogui.hotkey(*DISCORD_LEAVE_HOTKEY),
                self.one_shot_state,
                now,
            )
            self._draw_mode_text(frame, "Discord hotkey", (0, 165, 255))
        else:
            self._reset_motion_states()
            if not detected_gesture:
                self._reset_one_shot()

    def _handle_fist_volume(self, center, frame, width):
        self.fist_state["confidence_frames"] += 1
        center_x, center_y = center
        self.pinch_state["active"] = False

        if (
            not self.fist_state["active"]
            and self.fist_state["confidence_frames"] >= FIST_MIN_CONFIDENCE_FRAMES
        ):
            self.fist_state["active"] = True
            self.fist_state["anchor_y"] = center_y
            self.fist_state["last_step"] = 0

        if self.fist_state["active"]:
            self._adjust_volume_from_fist(center_y)
            cv2.circle(frame, (center_x, center_y), 14, (0, 255, 0), -1)
            cv2.line(
                frame,
                (0, self.fist_state["anchor_y"]),
                (width, self.fist_state["anchor_y"]),
                (0, 255, 255),
                2,
            )
            self._draw_mode_text(frame, "Fist volume mode", (0, 255, 0))

    def _handle_pinch_volume(self, center, frame, width):
        self.fist_state["active"] = False
        self.fist_state["confidence_frames"] = 0
        center_x, center_y = center

        if not self.pinch_state["active"]:
            self.pinch_state["active"] = True
            self.pinch_state["anchor_y"] = center_y
            self.pinch_state["last_step"] = 0

        self._adjust_player_volume_from_pinch(center_y)
        cv2.circle(frame, (center_x, center_y), 14, (255, 0, 255), -1)
        cv2.line(
            frame,
            (0, self.pinch_state["anchor_y"]),
            (width, self.pinch_state["anchor_y"]),
            (255, 0, 255),
            2,
        )
        self._draw_mode_text(frame, "Pinch arrow volume", (255, 0, 255))

    def _adjust_volume_from_fist(self, current_y):
        movement = current_y - self.fist_state["anchor_y"]
        current_step = int(movement / FIST_MOVE_STEP_PX)
        step_delta = current_step - self.fist_state["last_step"]

        if step_delta == 0:
            return

        key = "volumedown" if step_delta > 0 else "volumeup"
        pyautogui.press(key, presses=abs(step_delta), interval=VOLUME_PRESS_INTERVAL)
        self.fist_state["last_step"] = current_step

    def _adjust_player_volume_from_pinch(self, current_y):
        movement = current_y - self.pinch_state["anchor_y"]
        current_step = int(movement / PINCH_MOVE_STEP_PX)
        step_delta = current_step - self.pinch_state["last_step"]

        if step_delta == 0:
            return

        key = "down" if step_delta > 0 else "up"
        pyautogui.press(key, presses=abs(step_delta), interval=VOLUME_PRESS_INTERVAL)
        self.pinch_state["last_step"] = current_step

    def _toggle_gui_control(self):
        self.gui_control_active = not self.gui_control_active
        self.dashboard.set_control_surface_active(self.gui_control_active)
        if self.gui_control_active:
            self.dashboard.showMaximized()
            self.dashboard.raise_()
            self.dashboard.activateWindow()
        self._reset_motion_states()
        self._reset_one_shot()
        self.swipe_tracker.reset()

    def _toggle_dashboard_lock(self):
        action = (
            GestureAction.UNLOCK_PAGE
            if self.dashboard.page_manager.locked
            else GestureAction.LOCK_PAGE
        )
        self.dashboard.dispatch_gesture(action)

    def _dashboard_back_or_exit(self):
        if self.dashboard.page_manager.locked:
            self.dashboard.dispatch_gesture(GestureAction.BACK)
        else:
            self._toggle_gui_control()

    def _reset_motion_states(self):
        self.fist_state["active"] = False
        self.fist_state["confidence_frames"] = 0
        self.fist_state["anchor_y"] = None
        self.fist_state["last_step"] = 0
        self.pinch_state["active"] = False
        self.pinch_state["anchor_y"] = None
        self.pinch_state["last_step"] = 0

    def _reset_one_shot(self):
        self.one_shot_state["last_seen"] = None
        self.one_shot_state["frames"] = 0
        self.one_shot_state["triggered"] = False

    def _reset_open_palm(self):
        self.open_palm_state["last_seen"] = None
        self.open_palm_state["frames"] = 0
        self.open_palm_state["triggered"] = False

    def _draw_status(self, frame, detected_gesture, center, height):
        mode = "GUI MENU" if self.gui_control_active else "DESKTOP"
        cv2.putText(
            frame,
            f"Mode: {mode}",
            (20, height - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (78, 220, 255) if self.gui_control_active else (255, 255, 255),
            2,
        )

        if detected_gesture and center:
            cv2.putText(
                frame,
                f"Gesture: {detected_gesture}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _draw_mode_text(self, frame, text, color):
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )


def main():
    app = QApplication(sys.argv)
    dashboard = GestureDashboard()
    dashboard.showMaximized()

    controller = GestureController(dashboard)
    timer = QTimer()
    timer.timeout.connect(controller.process_frame)
    timer.start(15)
    app.aboutToQuit.connect(controller.shutdown)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

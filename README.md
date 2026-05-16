# GesturePY

Simple webcam-based gesture control using MediaPipe and OpenCV.

Prerequisites
- Python 3.8+
- A working webcam

Install

```bash
pip install -r requirements.txt
```

Run

```bash
python gesturesMain.py
```

Usage
- The script opens a window showing webcam feed and hand landmarks.
- Fist up/down changes the system volume.
- Open palm pauses or unpauses gesture control.
- Pinch up/down presses the up/down arrow keys for the focused video player.
- Pointing with one finger sends media play/pause.
- Thumbs down sends `Alt+F4` to close the focused window.
- Peace sign presses `Ctrl+Alt+Shift+D`. Set Discord's disconnect/leave-call keybind to that shortcut if you want this gesture to leave a call.
- Press `q` to quit.

Notes
- On some systems `pyautogui` may require additional OS permissions for media keys.
- For pinch volume, click/focus the browser video player first so arrow keys control its volume.

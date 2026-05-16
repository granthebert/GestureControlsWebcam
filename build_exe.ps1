$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt
python -m pip install pyinstaller

$mediaPipeDll = python -c "import mediapipe, pathlib; print(pathlib.Path(mediapipe.__file__).parent / 'tasks' / 'c' / 'libmediapipe.dll')"

python -m PyInstaller `
    --clean `
    --onefile `
    --name GesturePY `
    --hidden-import mediapipe.tasks.c `
    --add-binary "$mediaPipeDll;mediapipe/tasks/c" `
    --add-data "hand_landmarker.task;." `
    gesturesMain.py

Write-Host ""
Write-Host "Build complete: dist\GesturePY.exe"

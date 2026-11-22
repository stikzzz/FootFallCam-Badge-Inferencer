# FootfallCam Badge Inferencer

Run staff badge detection on an overhead video. The pipeline uses a fine-tuned YOLO26s person detector, ByteTrack for person IDs, and a fine-tuned YOLO26n badge detector. The repository includes both OpenVINO model exports.

## Windows setup

Install Python 3.12 and make FFmpeg available on PATH. Open PowerShell in the repository folder and set up the local environment once.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
ffmpeg -version
~~~

## Run on a video

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_badge_demo.ps1 "C:\Videos\new_test.mp4"
~~~

The launcher saves an annotated MP4 and a frame-by-frame CSV in `output/demo/`. Use `-DetectorDevice cpu` if the OpenVINO GPU backend is unavailable, or `-MaxFrames 100` for a short test. Bring your own input video. Once the environment is set up, the run uses the included local model files.

The CSV contains zero-based frame indices, timestamps, track IDs, person boxes, and badge boxes when a badge is detected. `badged=1` indicates that the track has accumulated three badge sightings within one second. The status follows the track ID as long as that ID remains active.

## Model attribution

The included model exports are fine-tuned Ultralytics YOLO26 models. Their metadata identifies the upstream license as AGPL-3.0. See [Ultralytics' license information](https://ultralytics.com/license) and the [upstream license text](ULTRALYTICS-LICENSE.txt).

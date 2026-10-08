#!/usr/bin/env bash
# Installs system/Python dependencies and downloads the models + fonts the pipeline uses.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo_if() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }
sudo_if apt-get install -y ffmpeg fluidsynth fluid-soundfont-gm libegl1 libgles2 >/dev/null
pip install -q numpy scipy opencv-python-headless pillow onnxruntime torch mediapipe mido pyloudnorm soundfile

mkdir -p models fonts
dl() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
dl https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl models/flownet_v4.25.pkl
dl https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx models/rvm_resnet50_fp32.onnx
dl https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task models/face_landmarker.task
dl https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth models/realesr-general-x4v3.pth
dl https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth models/realesr-general-wdn-x4v3.pth

GF=https://raw.githubusercontent.com/google/fonts/main/ofl
for f in "cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf" "cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf" \
         "instrumentserif/InstrumentSerif-Regular.ttf" "instrumentserif/InstrumentSerif-Italic.ttf" \
         "playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf" "playfairdisplay/PlayfairDisplay-Italic%5Bwght%5D.ttf" \
         "montserrat/Montserrat%5Bwght%5D.ttf" "manrope/Manrope%5Bwght%5D.ttf"; do
  name=$(basename "$f" | sed 's/%5B/[/g; s/%5D/]/g')
  dl "$GF/$f" "fonts/$name"
done
echo "setup done"

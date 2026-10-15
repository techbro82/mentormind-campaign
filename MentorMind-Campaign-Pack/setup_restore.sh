#!/bin/sh
# Restore the MentorMind campaign workspace in a fresh Claude sandbox.
# Run from inside the MentorMind-Campaign-Pack folder:  sh setup_restore.sh
set -e
mkdir -p /home/claude/social /home/claude/voice /tmp/fonts /home/claude/clips /home/claude/clips2 /home/claude/clips3
cp scripts/*.py /home/claude/social/
cp voice/* /home/claude/voice/
cp fonts/*.ttf /tmp/fonts/
cp configs/day01_real.json configs/day01_v2.json /home/claude/clips/
cp configs/day02.json /home/claude/clips2/
cp configs/day03.json /home/claude/clips3/
pip install --break-system-packages kokoro-onnx soundfile pocketsphinx opencv-python-headless pillow numpy scipy
# The voice model is too big to ship (about 340 MB). Download the FULL-SIZE model, not the int8 one (it garbles speech).
cd /home/claude/voice
[ -f kokoro-v1.0.onnx ] || curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
[ -f voices-v1.0.bin ] || curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
echo "Restored. Still needed for Days 1 to 3: unzip MentorMind-Day1-3-source-clips.zip into /home/claude/clips, clips2, clips3."

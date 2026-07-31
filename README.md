BELT is a friendly, funny, talking Robot that can hold conversations and gesture.



set up ~1.5 min:
python -m pip install \
    openai \
    joblib \
    "numpy==2.4.6" \
    scikit-learn \
    sentence-transformers \
    python-dotenv
python -m pip install -U sentence-transformers datasets accelerate

cv:
pip install insightface onnxruntime opencv-python joblib "numpy==2.4.6"
pip install ultralytics


connect to tp link
ssh -o ConnectTimeout=10 tina@192.168.0.56
ai_academy_14235156


run these commands:
conda deactivate
cd robo-voice
source ros_venv/bin/activate
source /opt/ros/jazzy/setup.bash

cd belt_v3

test Qwen Aiden on the robot:
python -m speech.belt_v3_speech_handle \
  "This is BELT testing the Qwen Aiden voice."


kill jobs
kill -9 $(jobs -p)
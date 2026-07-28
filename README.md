BELT is a friendly, funny, talking Robot that can hold conversations and gesture.



set up ~1.5 min:
python -m pip install \
    openai \
    joblib \
    numpy \
    scikit-learn \
    sentence-transformers \
    python-dotenv
python -m pip install -U sentence-transformers datasets accelerate

cv:
pip install insightface onnxruntime opencv-python joblib numpy
pip install pyttsx3 ultralytics


connect to tp link
ssh -o ConnectTimeout=10 tina@192.168.0.56
ai_academy_14235156


run these commands:
conda deactivate
cd robo-voice
source ros_venv/bin/activate
cd belt_v3

test speaking:
cd ~
python3.12 test_speak.py "hello"

maybe?
source ~/unitree_ros2/setup.sh
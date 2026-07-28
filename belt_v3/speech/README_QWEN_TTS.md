# BELT Qwen3-TTS audio

BELT generates speech with
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` on the computer running
`belt_v3_main.py`. It sends the complete generated WAV file as a
`std_msgs/msg/UInt8MultiArray` message on `/g1/audio/play`.

## Main-computer setup

Use an isolated Python environment and install:

```bash
pip install -r speech/requirements-qwen-tts.txt
```

The model weights download automatically the first time Qwen TTS loads and
remain in the Hugging Face cache. CUDA is strongly recommended. CPU fallback
is supported by the integration but can be too slow for interactive speech.

Select the voice with `VOICE` in `belt_v3_main.py`. Available voices are
`Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`,
`Ono_Anna`, and `Sohee`.

## ROS and robot setup

Before running BELT, source the Unitree ROS environment on the computer:

```bash
source /opt/unitree_ros2/setup.sh
```

The teacher-provided `stream_audio_bridge.py` must be running on the robot. It
subscribes to `/g1/audio/play` and forwards each WAV to the Unitree audio
service. Both machines must use the same `ROS_DOMAIN_ID` and
`CYCLONEDDS_URI`.

`speech/publish_wav.py` can also publish an existing WAV manually:

```bash
python3 -m speech.publish_wav /path/to/audio.wav
```

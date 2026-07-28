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

`belt_v3_main.py` preloads Qwen before accepting the first input. Wait for
`[TTS READY]` in the terminal; the model then remains in memory and is reused
for every response until BELT exits. The standalone
`python3 -m speech.belt_v3_speech_handle` command is a one-shot process, so a
new invocation of that command must load the model again.

Select the voice with `VOICE` in `belt_v3_main.py`. Available voices are
`Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`,
`Ono_Anna`, and `Sohee`.

BELT currently locks its default voice to `Aiden`. On startup, the terminal
prints the exact model, speaker, language, and generator source file. Before
each robot publication it also prints a generation record identifying
`speaker=Aiden`.

The most recently generated Aiden recording remains available for diagnosis:

```text
/tmp/belt_v3_generated_audio/last_qwen_aiden.wav
```

Playing that file locally lets you distinguish the generated Qwen voice from
anything else speaking on the computer or robot.

## ROS and robot setup

Before running BELT, source ROS 2 Jazzy on the computer:

```bash
source /opt/ros/jazzy/setup.bash
```

The teacher-provided `stream_audio_bridge.py` must be running on the robot. It
subscribes to `/g1/audio/play` and forwards each WAV to the Unitree audio
service. Both machines must use the same `ROS_DOMAIN_ID` and
`CYCLONEDDS_URI`.

`speech/publish_wav.py` can also publish an existing WAV manually:

```bash
python3 -m speech.publish_wav /path/to/audio.wav
```

To test the complete Aiden generation and robot-publication path without
starting the conversation loop:

```bash
python3 -m speech.belt_v3_speech_handle \
  "This is BELT testing the Qwen Aiden voice."
```

Both `belt_v3_main.py` and the computer-vision greeting programs use this same
speech handler. They do not use the operating system's `pyttsx3` voice.

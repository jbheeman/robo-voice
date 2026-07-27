# BELT Qwen3-TTS audio

BELT generates speech with
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` on the computer running
`belt_v3_main.py`. It sends the generated WAV and its text/voice metadata to
the robot over the ROS 2 topic `/belt/audio_file`.

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

## Robot setup

The robot needs ROS 2, `std_msgs`, and ALSA's `aplay`. From the `belt_v3`
directory, source ROS and run:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m speech.belt_v3_robot_audio_player
```

Start the robot audio player before `belt_v3_main.py`. Both machines must use
the same `ROS_DOMAIN_ID` and be able to discover each other over ROS 2.

The player stops the current `aplay` process when newer BELT audio arrives, so
an old response does not have to finish before a new one starts.

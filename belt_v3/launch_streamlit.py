#!/usr/bin/env python3
"""Launch a live webpage that logs BELT's heard and spoken text."""

from __future__ import annotations

import atexit
import csv
import importlib.util
import io
import json
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from speech.publish_wav import DEFAULT_TOPIC

INPUT_TOPIC = "/audio_msg_bridge"
OUTPUT_TOPIC = DEFAULT_TOPIC
TRANSCRIPT_SETTLE_SECONDS = 0.8
REPEATED_UTTERANCE_GAP_SECONDS = 1.5
MAX_LOG_ENTRIES = 500
PAGE_REFRESH_SECONDS = 1.0
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501
DASHBOARD_START_TIMEOUT_SECONDS = 8.0


class RobotAudioLog:
    """Collect input transcripts and output speech from ROS 2."""

    def __init__(self) -> None:
        self._entries: deque[dict[str, str]] = deque(maxlen=MAX_LOG_ENTRIES)
        self._entries_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status = "Starting ROS listener..."
        self._error: str | None = None

        self._pending_text: str | None = None
        self._pending_metadata: dict[str, Any] = {}
        self._pending_changed_at = 0.0
        self._last_input_message_at = 0.0
        self._last_logged_input: str | None = None

        self._context: Any | None = None
        self._node: Any | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_ros,
            name="belt-streamlit-ros-listener",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def _run_ros(self) -> None:
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import (
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from std_msgs.msg import String, UInt8MultiArray
        except ImportError as error:
            self._set_error(
                "ROS 2 Python packages are unavailable. Source "
                "/opt/ros/jazzy/setup.bash before launching this page."
            )
            return

        context = Context()
        self._context = context
        executor: Any | None = None

        try:
            rclpy.init(args=None, context=context)
            node = rclpy.create_node(
                "belt_streamlit_audio_log",
                context=context,
            )
            self._node = node
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)

            input_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )

            self._subscriptions = [
                node.create_subscription(
                    String,
                    INPUT_TOPIC,
                    self._input_callback,
                    input_qos,
                ),
                node.create_subscription(
                    UInt8MultiArray,
                    OUTPUT_TOPIC,
                    self._output_callback,
                    input_qos,
                ),
            ]

            self._set_status(
                f"Listening on {INPUT_TOPIC} and {OUTPUT_TOPIC}"
            )

            while context.ok() and not self._closed:
                executor.spin_once(timeout_sec=0.1)
                self._flush_stable_transcript()

        except Exception as error:
            if not self._closed:
                self._set_error(f"ROS listener stopped: {error}")
        finally:
            if self._pending_text is not None:
                self._flush_stable_transcript(force=True)

            if self._node is not None:
                if executor is not None:
                    executor.remove_node(self._node)
                    executor.shutdown()

                self._node.destroy_node()
                self._node = None

            if context.ok():
                context.shutdown()

    def _input_callback(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(payload, dict):
            return

        text = payload.get("text")
        if not isinstance(text, str):
            return

        text = text.strip()
        if not text or text in {".", "。"}:
            return

        now = time.monotonic()
        message_gap = (
            now - self._last_input_message_at
            if self._last_input_message_at
            else REPEATED_UTTERANCE_GAP_SECONDS
        )
        self._last_input_message_at = now

        # Ignore a relay repeatedly publishing the previously logged partial.
        # The same phrase is accepted again after an actual silence gap.
        if (
            text == self._last_logged_input
            and self._pending_text is None
            and message_gap < REPEATED_UTTERANCE_GAP_SECONDS
        ):
            return

        if text != self._pending_text:
            self._pending_text = text
            self._pending_changed_at = now
            self._pending_metadata = payload

        if payload.get("is_final") is True:
            self._flush_stable_transcript(force=True)

    def _flush_stable_transcript(self, force: bool = False) -> None:
        if self._pending_text is None:
            return

        if (
            not force
            and time.monotonic() - self._pending_changed_at
            < TRANSCRIPT_SETTLE_SECONDS
        ):
            return

        text = self._pending_text
        metadata = self._pending_metadata
        self._pending_text = None
        self._pending_metadata = {}
        self._last_logged_input = text

        details = []
        language = metadata.get("language")
        confidence = metadata.get("confidence")
        speaker_id = metadata.get("speaker_id")

        if language:
            details.append(f"language={language}")
        if confidence is not None:
            details.append(f"confidence={confidence}")
        if speaker_id is not None:
            details.append(f"speaker={speaker_id}")

        self._append_entry(
            direction="Input",
            text=text,
            topic=INPUT_TOPIC,
            details=", ".join(details),
        )

    def _output_callback(self, message: Any) -> None:
        byte_count = len(message.data)
        self._append_entry(
            direction="Output",
            text="WAV audio",
            topic=OUTPUT_TOPIC,
            details=f"{byte_count} bytes",
        )

    def _append_entry(
        self,
        *,
        direction: str,
        text: str,
        topic: str,
        details: str,
    ) -> None:
        timestamp = datetime.now().astimezone()
        entry = {
            "Time": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "Direction": direction,
            "Text": text,
            "Topic": topic,
            "Details": details,
        }

        with self._entries_lock:
            self._entries.append(entry)

    def snapshot(self) -> list[dict[str, str]]:
        with self._entries_lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._entries_lock:
            self._entries.clear()

    def status(self) -> tuple[str, str | None]:
        with self._status_lock:
            return self._status, self._error

    def _set_status(self, status: str) -> None:
        with self._status_lock:
            self._status = status
            self._error = None

    def _set_error(self, error: str) -> None:
        with self._status_lock:
            self._status = "Listener unavailable"
            self._error = error

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        context = self._context

        if context is not None and context.ok():
            context.shutdown()

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _entries_as_csv(entries: list[dict[str, str]]) -> str:
    output = io.StringIO()
    fieldnames = ["Time", "Direction", "Text", "Topic", "Details"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(entries)
    return output.getvalue()


def render_dashboard() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="BELT Audio Log",
        page_icon="🤖",
        layout="wide",
    )

    @st.cache_resource
    def get_audio_log() -> RobotAudioLog:
        return RobotAudioLog()

    audio_log = get_audio_log()

    st.title("BELT Audio Log")
    st.caption(
        "Live transcript input from the robot and text sent to robot speech."
    )

    left, _ = st.columns([1, 4])
    with left:
        if st.button("Clear log", use_container_width=True):
            audio_log.clear()
            st.rerun()

    def render_live_log() -> None:
        status, error = audio_log.status()
        if error:
            st.error(error)
        else:
            st.success(status)

        entries = audio_log.snapshot()
        input_count = sum(
            entry["Direction"] == "Input" for entry in entries
        )
        output_count = sum(
            entry["Direction"] == "Output" for entry in entries
        )

        input_metric, output_metric, total_metric = st.columns(3)
        input_metric.metric("Inputs heard", input_count)
        output_metric.metric("Outputs spoken", output_count)
        total_metric.metric("Total events", len(entries))

        if entries:
            st.dataframe(
                entries,
                hide_index=True,
                use_container_width=True,
                column_order=["Time", "Direction", "Text", "Details"],
            )
        else:
            st.info("Waiting for the robot to hear or say something...")

        st.download_button(
            "Download CSV",
            data=_entries_as_csv(entries),
            file_name="belt_audio_log.csv",
            mime="text/csv",
            disabled=not entries,
        )

    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        fragment(run_every=PAGE_REFRESH_SECONDS)(render_live_log)()
    else:
        st.warning(
            "This Streamlit version does not support automatic refresh. "
            "Reload the page to see new events."
        )
        render_live_log()


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False

    try:
        return get_script_run_ctx(suppress_warning=True) is not None
    except TypeError:
        return get_script_run_ctx() is not None


def _streamlit_command(extra_args: list[str] | None = None) -> list[str]:
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(
            "Streamlit is not installed. Install it with: "
            "python3 -m pip install streamlit"
        )

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        DASHBOARD_HOST,
        "--server.port",
        str(DASHBOARD_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    if extra_args:
        command.extend(extra_args)

    return command


def start_streamlit(
    extra_args: list[str] | None = None,
) -> subprocess.Popen[bytes]:
    """Start the in-memory dashboard without blocking BELT's main loop."""
    if _dashboard_port_is_open():
        raise RuntimeError(
            f"Port {DASHBOARD_PORT} is already in use. Stop the existing "
            "dashboard before starting BELT."
        )

    process = subprocess.Popen(_streamlit_command(extra_args))
    deadline = time.monotonic() + DASHBOARD_START_TIMEOUT_SECONDS

    while not _dashboard_port_is_open():
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "The BELT audio dashboard exited during startup "
                f"with status {return_code}."
            )

        if time.monotonic() >= deadline:
            stop_streamlit(process)
            raise RuntimeError(
                "The BELT audio dashboard did not become ready within "
                f"{DASHBOARD_START_TIMEOUT_SECONDS:.1f} seconds."
            )

        time.sleep(0.1)

    atexit.register(stop_streamlit, process)
    print(
        "BELT audio dashboard ready at "
        f"http://192.168.0.56:{DASHBOARD_PORT}"
    )
    return process


def _dashboard_port_is_open() -> bool:
    try:
        with socket.create_connection(
            ("127.0.0.1", DASHBOARD_PORT),
            timeout=0.2,
        ):
            return True
    except OSError:
        return False


def stop_streamlit(process: subprocess.Popen[bytes] | None) -> None:
    """Stop a dashboard process created by ``start_streamlit``."""
    if process is None or process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def launch_streamlit() -> int:
    """Run the dashboard until it is stopped when this file is executed."""
    try:
        process = start_streamlit(sys.argv[1:])
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    try:
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        stop_streamlit(process)


if __name__ == "__main__":
    if _running_inside_streamlit():
        render_dashboard()
    else:
        raise SystemExit(launch_streamlit())

"""
BELT Unitree G1 movement + gesture executor.

Default mode is DRY RUN:
- It does not move the real robot.
- It only prints what the robot would do.
"""

import argparse
import json
import time
from dataclasses import dataclass


ALLOWED_ACTIONS = {
    "stop",
    "move_forward",
    "turn_left",
    "turn_right",
    "wave",
    "handshake",
}


MOVE_ACTIONS = {
    "move_forward",
    "turn_left",
    "turn_right",
}


GESTURE_ACTIONS = {
    "wave",
    "handshake",
}


MAX_MOVE_DURATION = 2.0
MAX_GESTURE_DURATION = 5.0

MAX_VX = 0.12
MAX_VY = 0.00
MAX_YAW_RATE = 0.20

CONTROL_PERIOD = 0.20


@dataclass
class RobotState:
    emergency_stop: bool = False
    blocked_by_cv: bool = False
    robot_ready: bool = True


@dataclass
class BeltAction:
    action_type: str
    duration: float = 0.0


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


class BeltUnitreeExecutor:
    def __init__(self, real_robot=False, iface=None, send_start=False):
        self.real_robot = real_robot
        self.iface = iface
        self.send_start = send_start
        self.state = RobotState()
        self.client = None

    def connect(self):
        """     
        Connecting to the robot
        """
        if not self.real_robot:
            print("[DRY RUN] Not connecting to the real robot.")
            return

        if not self.iface:
            raise ValueError("Real robot mode needs --iface, like --iface eth0")

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        print(f"[REAL] Connecting to Unitree G1 using interface: {self.iface}")

        ChannelFactoryInitialize(0, self.iface)

        self.client = LocoClient()
        self.client.SetTimeout(10.0)
        self.client.Init()

        if self.send_start:
            print("[REAL] Sending Start() command.")
            self.client.Start()
            time.sleep(0.5)

        self.stop()

    def parse_llm_output(self, llm_json_text):
        """
        Convert LLM JSON text into speech text and a BeltAction.

        What I think the format should look like if we are telling it like what to do:
        {
            "speak": "Follow me.",
            "action": {
                "type": "move_forward",
                "duration": 1.5
            }
        }
        """

        try:
            data = json.loads(llm_json_text)
        except json.JSONDecodeError:
            print("Invalid JSON from LLM. Robot will stop.")
            return "", BeltAction("stop", 0.0)

        speak_text = data.get("speak", "")

        action_data = data.get("action", {})
        action_type = action_data.get("type", "stop")
        duration = action_data.get("duration", 0.0)

        try:
            duration = float(duration)
        except (ValueError, TypeError):
            duration = 0.0

        return speak_text, BeltAction(action_type, duration)

    def validate_action(self, action):
        """
        Safety check before anything reaches the robot.
        """

        if action.action_type not in ALLOWED_ACTIONS:
            print(f"Blocked unsafe/unknown action: {action.action_type}")
            return BeltAction("stop", 0.0)

        if self.state.emergency_stop:
            print("Emergency stop is active.")
            return BeltAction("stop", 0.0)

        if not self.state.robot_ready:
            print("Robot is not ready.")
            return BeltAction("stop", 0.0)

        if self.state.blocked_by_cv and action.action_type in MOVE_ACTIONS:
            print("CV says path is blocked, so movement is stopped.")
            return BeltAction("stop", 0.0)

        if action.action_type in MOVE_ACTIONS:
            safe_duration = min(action.duration, MAX_MOVE_DURATION)
            safe_duration = max(safe_duration, 0.0)
            return BeltAction(action.action_type, safe_duration)

        if action.action_type in GESTURE_ACTIONS:
            safe_duration = min(action.duration, MAX_GESTURE_DURATION)
            safe_duration = max(safe_duration, 0.0)
            return BeltAction(action.action_type, safe_duration)

        return action

    def execute_from_llm_json(self, llm_json_text):
        """
        makinf sure things are safe before executing the action
        """
        speak_text, requested_action = self.parse_llm_output(llm_json_text)
        safe_action = self.validate_action(requested_action)

        print("\n==============================")
        print("BELT UNITREE EXECUTOR")
        print("==============================")
        print(f"Speak text: {speak_text}")
        print(f"Requested action: {requested_action.action_type}")
        print(f"Requested duration: {requested_action.duration}")
        print(f"Safe action: {safe_action.action_type}")
        print(f"Safe duration: {safe_action.duration}")

        self.execute_action(safe_action)

    def execute_action(self, action):
        if action.action_type == "stop":
            self.stop()

        elif action.action_type == "move_forward":
            self.move_forward(action.duration)

        elif action.action_type == "turn_left":
            self.turn_left(action.duration)

        elif action.action_type == "turn_right":
            self.turn_right(action.duration)

        elif action.action_type == "wave":
            self.wave(action.duration)

        elif action.action_type == "handshake":
            self.handshake(action.duration)

    def send_velocity_once(self, vx, vy, yaw_rate, duration):
        """
        Send one velocity command.

        vx = forward/backward
        vy = sideways
        yaw_rate = turning
        """

        vx = clamp(vx, -MAX_VX, MAX_VX)
        vy = clamp(vy, -MAX_VY, MAX_VY)
        yaw_rate = clamp(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE)

        duration = clamp(duration, 0.05, 0.30)

        if not self.real_robot:
            print(
                f"[DRY RUN] Unitree SetVelocity("
                f"vx={vx:.2f}, vy={vy:.2f}, yaw_rate={yaw_rate:.2f}, "
                f"duration={duration:.2f})"
            )
            return

        if self.client is None:
            raise RuntimeError("Unitree client is not connected.")

        code = self.client.SetVelocity(vx, vy, yaw_rate, duration)

        if code != 0:
            print(f"[WARN] SetVelocity returned code: {code}")

    def move_for_duration(self, vx, vy, yaw_rate, duration):
        """
        Move for a short duration by refreshing small velocity commands.
        Always stop afterward.
        """

        duration = max(0.0, duration)
        end_time = time.monotonic() + duration

        try:
            while time.monotonic() < end_time:
                remaining = end_time - time.monotonic()
                command_duration = min(CONTROL_PERIOD + 0.05, remaining + 0.05)

                self.send_velocity_once(vx, vy, yaw_rate, command_duration)

                sleep_time = min(CONTROL_PERIOD, max(0.01, remaining))
                time.sleep(sleep_time)

        finally:
            self.stop()

    def stop(self):
        print("Robot STOP command.")
        print("Unitree command: vx=0.00, vy=0.00, yaw_rate=0.00")

        if not self.real_robot:
            return

        if self.client is None:
            return

        for _ in range(3):
            self.client.SetVelocity(0.0, 0.0, 0.0, 0.20)
            time.sleep(0.05)

    def move_forward(self, duration):
        print(f"Robot moving forward for {duration} seconds.")
        self.move_for_duration(
            vx=0.12,
            vy=0.00,
            yaw_rate=0.00,
            duration=duration,
        )

    def turn_left(self, duration):
        print(f"Robot turning left for {duration} seconds.")
        self.move_for_duration(
            vx=0.00,
            vy=0.00,
            yaw_rate=0.20,
            duration=duration,
        )

    def turn_right(self, duration):
        print(f"Robot turning right for {duration} seconds.")
        self.move_for_duration(
            vx=0.00,
            vy=0.00,
            yaw_rate=-0.20,
            duration=duration,
        )

    def wave(self, duration):
        print(f"Robot waving for about {duration} seconds.")

        if not self.real_robot:
            print("[DRY RUN] Unitree gesture: WaveHand()")
            return

        if self.client is None:
            raise RuntimeError("Unitree client is not connected.")

        self.client.WaveHand()
        time.sleep(min(duration, MAX_GESTURE_DURATION))

    def handshake(self, duration):
        print(f"Robot doing handshake gesture for about {duration} seconds.")

        if not self.real_robot:
            print("[DRY RUN] Unitree gesture: ShakeHand()")
            return

        if self.client is None:
            raise RuntimeError("Unitree client is not connected.")

        self.client.ShakeHand()
        time.sleep(min(duration, MAX_GESTURE_DURATION))
        self.client.ShakeHand()


def make_llm_json(action_type, duration):
    """
    Make fake LLM JSON for testing one action.
    Later, your real BELT LLM step will produce this JSON.
    """

    return json.dumps(
        {
            "speak": f"Executing action: {action_type}",
            "action": {
                "type": action_type,
                "duration": duration,
            },
        }
    )


DRY_RUN_DEMO_OUTPUTS = [
    """
    {
        "speak": "Hi, I am BELT. Nice to meet you.",
        "action": {
            "type": "wave",
            "duration": 2.0
        }
    }
    """,
    """
    {
        "speak": "Follow me. I will move forward.",
        "action": {
            "type": "move_forward",
            "duration": 1.5
        }
    }
    """,
    """
    {
        "speak": "I will turn left now.",
        "action": {
            "type": "turn_left",
            "duration": 1.0
        }
    }
    """,
    """
    {
        "speak": "I will turn right now.",
        "action": {
            "type": "turn_right",
            "duration": 1.0
        }
    }
    """,
    """
    {
        "speak": "This action asks for too long, so the safety code will shorten it.",
        "action": {
            "type": "move_forward",
            "duration": 99.0
        }
    }
    """,
    """
    {
        "speak": "This is an unsafe unknown action.",
        "action": {
            "type": "run_fast",
            "duration": 5.0
        }
    }
    """,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="BELT Unitree G1 movement and gesture executor"
    )

    parser.add_argument(
        "--real",
        action="store_true",
        help="Actually send commands to the real Unitree robot",
    )

    parser.add_argument(
        "--iface",
        default=None,
        help="Network interface connected to robot, like eth0 or enp2s0",
    )

    parser.add_argument(
        "--send-start",
        action="store_true",
        help="Optionally send Unitree Start() after connecting",
    )

    parser.add_argument(
        "--action",
        default="demo",
        choices=[
            "demo",
            "stop",
            "move_forward",
            "turn_left",
            "turn_right",
            "wave",
            "handshake",
        ],
        help="Action to test",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration for the selected action",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    executor = BeltUnitreeExecutor(
        real_robot=args.real,
        iface=args.iface,
        send_start=args.send_start,
    )

    if args.action == "demo":
        if args.real:
            print("Real robot mode does not run the full demo sequence.")
            print("Choose one action, for example:")
            print("python movement_unitree.py --real --iface YOUR_INTERFACE --action stop")
            return

        for output in DRY_RUN_DEMO_OUTPUTS:
            executor.execute_from_llm_json(output)

        return

    if args.real:
        print("\nREAL ROBOT SAFETY CHECK")
        print("1. A mentor/instructor is present.")
        print("2. The robot is on a flat, clear area.")
        print("3. Someone has the robot remote or emergency stop ready.")
        print("4. You are testing only one short action.")

        typed = input('Type exactly "RUN" to continue: ')

        if typed != "RUN":
            print("Cancelled.")
            return

        executor.connect()
    else:
        executor.connect()

    try:
        test_json = make_llm_json(args.action, args.duration)
        executor.execute_from_llm_json(test_json)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping robot.")

    finally:
        executor.stop()


if __name__ == "__main__":
    main()

"""
BELT Unitree G1 movement + gesture executor.

This file starts in DRY RUN mode, which means it only prints what the robot
would do. Use --real only when the robot is actually ready to be tested.
"""

import argparse
import json
import time


# Keep the robot slow and give every action a time limit.
MAX_MOVE_TIME = 2.0
MAX_GESTURE_TIME = 5.0
MAX_FORWARD_SPEED = 0.12
MAX_TURN_SPEED = 0.20
CONTROL_PERIOD = 0.20


# These are the only movement commands the LLM is allowed to request.
MOVEMENT_COMMANDS = {
    "move_forward": {"vx": 0.12, "yaw_rate": 0.00},
    "turn_left": {"vx": 0.00, "yaw_rate": 0.20},
    "turn_right": {"vx": 0.00, "yaw_rate": -0.20},
}

# These names match the Unitree SDK gesture methods.
GESTURE_COMMANDS = {
    "wave": "WaveHand",
    "handshake": "ShakeHand",
}

ALLOWED_ACTIONS = {"stop", *MOVEMENT_COMMANDS.keys(), *GESTURE_COMMANDS.keys()}


class RobotState:
    """Small place to keep safety flags that other code can update."""

    def __init__(self):
        self.emergency_stop = False
        self.blocked_by_cv = False
        self.robot_ready = True


def clamp(value, low, high):
    """Keep a number inside a safe range."""
    return max(low, min(value, high))


def make_llm_json(action_type, duration, speak=None):
    """Make fake LLM output so we can test this file without the real LLM."""
    return json.dumps(
        {
            "speak": speak or f"Executing action: {action_type}",
            "action": {
                "type": action_type,
                "duration": duration,
            },
        }
    )


class BeltUnitreeExecutor:
    def __init__(self, real_robot=False, iface=None, send_start=False):
        self.real_robot = real_robot
        self.iface = iface
        self.send_start = send_start
        self.client = None
        self.state = RobotState()

    def connect(self):
        """Connect to the robot only when real robot mode is turned on."""
        if not self.real_robot:
            print("[DRY RUN] Not connecting to the real robot.")
            return

        if not self.iface:
            raise ValueError("Real robot mode needs --iface, like --iface eth0")

        # Import this here so dry-run mode still works on computers without the SDK.
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
        Read the LLM response.

        Expected format:
        {
            "speak": "Follow me.",
            "action": {"type": "move_forward", "duration": 1.5}
        }
        """
        try:
            data = json.loads(llm_json_text)
        except json.JSONDecodeError:
            print("The LLM response was not valid JSON, so the robot will stop.")
            return "", "stop", 0.0

        speak_text = data.get("speak", "")
        action_info = data.get("action", {})
        action_type = action_info.get("type", "stop")
        duration = action_info.get("duration", 0.0)

        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = 0.0

        return speak_text, action_type, duration

    def make_action_safe(self, action_type, duration):
        """Check the action before we let it reach the robot."""
        if action_type not in ALLOWED_ACTIONS:
            print(f"Blocked unknown action: {action_type}")
            return "stop", 0.0

        if self.state.emergency_stop:
            print("Emergency stop is active.")
            return "stop", 0.0

        if not self.state.robot_ready:
            print("Robot is not ready.")
            return "stop", 0.0

        if self.state.blocked_by_cv and action_type in MOVEMENT_COMMANDS:
            print("The camera says the path is blocked, so movement is stopped.")
            return "stop", 0.0

        if action_type in MOVEMENT_COMMANDS:
            return action_type, clamp(duration, 0.0, MAX_MOVE_TIME)

        if action_type in GESTURE_COMMANDS:
            return action_type, clamp(duration, 0.0, MAX_GESTURE_TIME)

        return "stop", 0.0

    # This keeps the old method name, but the code inside is simpler now.
    def validate_action(self, action_type, duration=None):
        if duration is None and hasattr(action_type, "action_type"):
            return self.make_action_safe(action_type.action_type, action_type.duration)
        return self.make_action_safe(action_type, duration)

    def execute_from_llm_json(self, llm_json_text):
        """Read the LLM output, safety-check it, then run the safe action."""
        speak_text, requested_action, requested_duration = self.parse_llm_output(llm_json_text)
        safe_action, safe_duration = self.make_action_safe(requested_action, requested_duration)

        print("\n==============================")
        print("BELT UNITREE EXECUTOR")
        print("==============================")
        print(f"Speak text: {speak_text}")
        print(f"Requested action: {requested_action}")
        print(f"Requested duration: {requested_duration}")
        print(f"Safe action: {safe_action}")
        print(f"Safe duration: {safe_duration}")

        self.execute_action(safe_action, safe_duration)

    def execute_action(self, action_type, duration=0.0):
        """Send the action to the right helper function."""
        if action_type == "stop":
            self.stop()
        elif action_type in MOVEMENT_COMMANDS:
            self.move(action_type, duration)
        elif action_type in GESTURE_COMMANDS:
            self.gesture(action_type, duration)
        else:
            print(f"I do not know how to run {action_type}, so I am stopping.")
            self.stop()

    def send_velocity_once(self, vx, yaw_rate, duration):
        """Send one short movement command."""
        vx = clamp(vx, -MAX_FORWARD_SPEED, MAX_FORWARD_SPEED)
        yaw_rate = clamp(yaw_rate, -MAX_TURN_SPEED, MAX_TURN_SPEED)
        duration = clamp(duration, 0.05, 0.30)

        if not self.real_robot:
            print(
                f"[DRY RUN] Unitree SetVelocity("
                f"vx={vx:.2f}, vy=0.00, yaw_rate={yaw_rate:.2f}, "
                f"duration={duration:.2f})"
            )
            return

        if self.client is None:
            raise RuntimeError("Unitree client is not connected.")

        code = self.client.SetVelocity(vx, 0.0, yaw_rate, duration)
        if code != 0:
            print(f"[WARN] SetVelocity returned code: {code}")

    def move(self, action_type, duration):
        """Move for a short time, then always stop."""
        command = MOVEMENT_COMMANDS[action_type]
        print(f"Robot will {action_type.replace('_', ' ')} for {duration} seconds.")

        end_time = time.monotonic() + max(duration, 0.0)

        try:
            while time.monotonic() < end_time:
                time_left = end_time - time.monotonic()
                command_time = min(CONTROL_PERIOD + 0.05, time_left + 0.05)

                self.send_velocity_once(
                    vx=command["vx"],
                    yaw_rate=command["yaw_rate"],
                    duration=command_time,
                )

                time.sleep(min(CONTROL_PERIOD, max(0.01, time_left)))
        finally:
            # This is important: the robot should not keep moving after the action ends.
            self.stop()

    def stop(self):
        """Stop the robot."""
        print("Robot STOP command.")
        print("Unitree command: vx=0.00, vy=0.00, yaw_rate=0.00")

        if not self.real_robot or self.client is None:
            return

        for _ in range(3):
            self.client.SetVelocity(0.0, 0.0, 0.0, 0.20)
            time.sleep(0.05)

    # These small methods make it easy to call actions directly from other files.
    def move_forward(self, duration):
        self.move("move_forward", duration)

    def turn_left(self, duration):
        self.move("turn_left", duration)

    def turn_right(self, duration):
        self.move("turn_right", duration)

    def wave(self, duration):
        self.gesture("wave", duration)

    def handshake(self, duration):
        self.gesture("handshake", duration)

    def gesture(self, action_type, duration):
        """Run a simple arm gesture."""
        method_name = GESTURE_COMMANDS[action_type]
        print(f"Robot will do the {action_type} gesture for about {duration} seconds.")

        if not self.real_robot:
            print(f"[DRY RUN] Unitree gesture: {method_name}()")
            return

        if self.client is None:
            raise RuntimeError("Unitree client is not connected.")

        gesture_method = getattr(self.client, method_name)
        gesture_method()
        time.sleep(clamp(duration, 0.0, MAX_GESTURE_TIME))

        # The original handshake called ShakeHand twice, so this keeps that behavior.
        if action_type == "handshake":
            gesture_method()


def build_demo_outputs():
    """A few safe dry-run examples to make sure everything still works."""
    return [
        make_llm_json("wave", 2.0, "Hi, I am BELT. Nice to meet you."),
        make_llm_json("move_forward", 1.5, "Follow me. I will move forward."),
        make_llm_json("turn_left", 1.0, "I will turn left now."),
        make_llm_json("turn_right", 1.0, "I will turn right now."),
        make_llm_json(
            "move_forward",
            99.0,
            "This is too long, so the safety code will shorten it.",
        ),
        make_llm_json("run_fast", 5.0, "This is an unknown action."),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="BELT Unitree G1 movement and gesture executor"
    )
    parser.add_argument("--real", action="store_true", help="Send commands to the real robot")
    parser.add_argument("--iface", default=None, help="Robot network interface, like eth0")
    parser.add_argument("--send-start", action="store_true", help="Send Unitree Start() after connecting")
    parser.add_argument(
        "--action",
        default="demo",
        choices=["demo", *sorted(ALLOWED_ACTIONS)],
        help="Action to test",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="How long the action should last")
    return parser.parse_args()


def confirm_real_robot_test():
    """One last check before sending anything to the real robot."""
    print("\nREAL ROBOT SAFETY CHECK")
    print("1. A mentor/instructor is present.")
    print("2. The robot is on a flat, clear area.")
    print("3. Someone has the remote or emergency stop ready.")
    print("4. You are testing one short action only.")

    return input('Type exactly "RUN" to continue: ') == "RUN"


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
            print("Try one action instead, like:")
            print("python movement_unitree.py --real --iface YOUR_INTERFACE --action stop")
            return

        executor.connect()
        for output in build_demo_outputs():
            executor.execute_from_llm_json(output)
        return

    if args.real and not confirm_real_robot_test():
        print("Cancelled.")
        return

    try:
        executor.connect()
        test_json = make_llm_json(args.action, args.duration)
        executor.execute_from_llm_json(test_json)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping robot.")
    finally:
        executor.stop()


if __name__ == "__main__":
    main()

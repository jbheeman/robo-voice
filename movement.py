"""
BELT movement + gesture dry-run executor.

This file tests the safe action system for the robot.

It does NOT move the real robot yet.
It only prints what the robot would do.

Later, we will connect these actions to the Unitree G1 SDK.
"""

import json
from dataclasses import dataclass


# These are the only actions BELT is allowed to request for now.
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


@dataclass
class RobotState:
    emergency_stop: bool = False
    blocked_by_cv: bool = False
    robot_ready: bool = True


@dataclass
class BeltAction:
    action_type: str
    duration: float = 0.0


class BeltMovementExecutor:
    def __init__(self):
        self.state = RobotState()

    def parse_llm_output(self, llm_json_text):
        """
        Convert the LLM JSON text into a BeltAction.

        Expected format:

        {
            "speak": "Hello, I am BELT.",
            "action": {
                "type": "wave",
                "duration": 2.0
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
        Safety check before the robot does anything.
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
        Main function:
        1. Read LLM JSON
        2. Validate action
        3. Execute safe action
        """

        speak_text, requested_action = self.parse_llm_output(llm_json_text)
        safe_action = self.validate_action(requested_action)

        print("\n==============================")
        print("BELT ACTION EXECUTOR")
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

    def stop(self):
        print("Robot would STOP.")
        print("Future Unitree command: vx=0.00, vy=0.00, yaw_rate=0.00")

    def move_forward(self, duration):
        print(f"Robot would move forward for {duration} seconds.")
        print("Future Unitree command: vx=0.12, vy=0.00, yaw_rate=0.00")

    def turn_left(self, duration):
        print(f"Robot would turn left for {duration} seconds.")
        print("Future Unitree command: vx=0.00, vy=0.00, yaw_rate=0.20")

    def turn_right(self, duration):
        print(f"Robot would turn right for {duration} seconds.")
        print("Future Unitree command: vx=0.00, vy=0.00, yaw_rate=-0.20")

    def wave(self, duration):
        print(f"Robot would wave for about {duration} seconds.")
        print("Future Unitree gesture: WaveHand()")

    def handshake(self, duration):
        print(f"Robot would do a handshake for about {duration} seconds.")
        print("Future Unitree gesture: ShakeHand()")


def main():
    executor = BeltMovementExecutor()

    # Fake LLM outputs for testing.
    # These imitate what your BELT LLM step might return later.
    test_outputs = [
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
        """
    ]

    for output in test_outputs:
        executor.execute_from_llm_json(output)


if __name__ == "__main__":
    main()


import time

class RobotMovementController:
    def __init__(self):
        # Placeholder for the SDK initialization of our robot
        self.robot = UnitreeRobotInterface() #TBA
        print("Robot movement controller initialized.")

    def execute_grasp(self, object):
        # 1. Compute Inverse Kinematics for arm joints
        # 2. Extend arm
        # 3. Close gripper
        # 4. Return arm to home position
        time.sleep(2)

    def track_and_approach(self, target_object, frame_width=640):
        # Commands the robot to approach the identified item
        depth = target_object["depth"]
        cx = target_object["center_x"]
        label = target_object["label"]
        
        center_pool = frame_width / 2
        offset_x = cx - center_pool
        
        print(f"[Movement] Target: {label} | Depth: {depth:.2f}m | Pixel Offset: {offset_x}")
        if depth > 0.4:
            # Object is far away -> Walk forward
            yaw_speed = 0.1 if offset_x > 50 else (-0.1 if offset_x < -50 else 0.0)
            forward_speed = 0.2  # unit is meters per second

        else:
            self.execute_grasp(target_object)

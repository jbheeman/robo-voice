import ast
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def _stub_module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


class BeltV3MainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simple_action_handle = Mock()
        cls.speech_handle = Mock()
        cls.testing_speech_handle = Mock()
        cls.preload_qwen_model = Mock()
        cls.tts_configuration_summary = Mock(return_value="TTS test config")
        cls.navigation_handle = Mock()
        cls.remember_conversation_turn = Mock()
        cls.compose_response = Mock()
        cls.prepare_vision_context_for_llm = Mock(return_value=None)
        cls.get_input = Mock()
        cls.terminal_get_input = Mock()
        cls.start_streamlit = Mock()
        cls.stop_streamlit = Mock()
        cls.close_cv = Mock()
        cls.get_cv_state = Mock()

        stub_modules = {
            "movement.belt_v3_simple_action_handle": _stub_module(
                "movement.belt_v3_simple_action_handle",
                simple_action_handle=cls.simple_action_handle,
            ),
            "speech.belt_v3_speech_handle": _stub_module(
                "speech.belt_v3_speech_handle",
                speech_handle=cls.speech_handle,
                testing_speech_handle=cls.testing_speech_handle,
            ),
            "speech.belt_v3_qwen_tts": _stub_module(
                "speech.belt_v3_qwen_tts",
                preload_qwen_model=cls.preload_qwen_model,
                tts_configuration_summary=cls.tts_configuration_summary,
            ),
            "navigation.belt_v3_navigation_handle": _stub_module(
                "navigation.belt_v3_navigation_handle",
                navigation_handle=cls.navigation_handle,
            ),
            "belt_v3_api": _stub_module(
                "belt_v3_api",
                ConversationMessage=dict,
                remember_conversation_turn=cls.remember_conversation_turn,
            ),
            "belt_v3_helper": _stub_module(
                "belt_v3_helper",
                compose_response=cls.compose_response,
                prepare_vision_context_for_llm=(
                    cls.prepare_vision_context_for_llm
                ),
            ),
            "belt_v3_input": _stub_module(
                "belt_v3_input",
                get_input=cls.get_input,
                terminal_get_input=cls.terminal_get_input,
            ),
            "launch_streamlit": _stub_module(
                "launch_streamlit",
                start_streamlit=cls.start_streamlit,
                stop_streamlit=cls.stop_streamlit,
            ),
            "comp_vision.belt_v3_cv": _stub_module(
                "comp_vision.belt_v3_cv",
                close_cv=cls.close_cv,
                get_cv_state=cls.get_cv_state,
            ),
        }

        with patch.dict(sys.modules, stub_modules):
            sys.modules.pop("belt_v3_main", None)
            cls.main_module = importlib.import_module("belt_v3_main")

    def setUp(self) -> None:
        self.main_module.DEBUG = False
        self.main_module.LAUNCH_STREAMLIT = False
        self.main_module.conversation.clear()

        for dependency in (
            self.simple_action_handle,
            self.speech_handle,
            self.testing_speech_handle,
            self.preload_qwen_model,
            self.navigation_handle,
            self.remember_conversation_turn,
            self.compose_response,
            self.prepare_vision_context_for_llm,
            self.get_input,
            self.terminal_get_input,
            self.start_streamlit,
            self.stop_streamlit,
            self.close_cv,
            self.get_cv_state,
        ):
            dependency.reset_mock()
            dependency.side_effect = None

    def test_terminal_main_loop_processes_one_complete_response(self) -> None:
        response = {
            "simple_action": {
                "requested": False,
                "actions": [],
            },
            "navigation": {
                "requested": False,
                "locations": [],
            },
            "speech": "Hello!",
        }
        self.main_module.USING_ROBOT = False
        self.terminal_get_input.side_effect = [
            "Hi, Belt",
            KeyboardInterrupt,
        ]
        self.compose_response.return_value = (response, [])

        self.main_module.main()

        self.compose_response.assert_called_once()
        self.testing_speech_handle.assert_called_once_with(
            "Hello!",
            self.main_module.VOICE,
        )
        self.remember_conversation_turn.assert_called_once_with(
            self.main_module.conversation,
            "Hi, Belt",
            "Hello!",
        )
        self.stop_streamlit.assert_called_once_with(None)
        self.close_cv.assert_called_once_with()

    def test_robot_main_loop_passes_disabled_wake_word_setting(self) -> None:
        response = {
            "simple_action": {
                "requested": False,
                "actions": [],
            },
            "navigation": {
                "requested": False,
                "locations": [],
            },
            "speech": "Hello!",
        }
        self.main_module.USING_ROBOT = True
        self.main_module.BELT_WAKE_WORD = False
        self.get_input.side_effect = [
            "Hello",
            KeyboardInterrupt,
        ]
        self.get_cv_state.return_value = None
        self.compose_response.return_value = (response, [])

        self.main_module.main()

        self.get_input.assert_any_call(require_wake_word=False)
        self.speech_handle.assert_called_once_with(
            "Hello!",
            self.main_module.VOICE,
        )

    def test_execute_modules_runs_speech_actions_and_navigation(self) -> None:
        self.main_module.USING_ROBOT = True
        response = {
            "simple_action": {
                "requested": True,
                "actions": ["wave"],
            },
            "navigation": {
                "requested": True,
                "locations": ["2110"],
            },
            "speech": "I can help with that.",
        }
        timings = {"output_audio": 0.0}

        self.main_module.execute_modules(response, timings)

        self.speech_handle.assert_called_once_with(
            "I can help with that.",
            self.main_module.VOICE,
        )
        self.simple_action_handle.assert_called_once_with(["wave"])
        self.navigation_handle.assert_called_once_with(["2110"])
        self.assertGreaterEqual(timings["output_audio"], 0.0)

    def test_response_pipeline_has_no_anonymous_functions(self) -> None:
        belt_v3_directory = Path(__file__).resolve().parent

        for filename in (
            "belt_v3_main.py",
            "belt_v3_helper.py",
            "belt_v3_api.py",
        ):
            source_path = belt_v3_directory / filename
            syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
            anonymous_functions = [
                node
                for node in ast.walk(syntax_tree)
                if isinstance(node, ast.Lambda)
            ]
            self.assertEqual(
                anonymous_functions,
                [],
                f"{filename} contains an anonymous function",
            )


if __name__ == "__main__":
    unittest.main()

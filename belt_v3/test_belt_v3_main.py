import ast
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


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
        cls.call_llm = Mock()
        cls.remember_conversation_turn = Mock()
        cls.compose_response = Mock()
        cls.prepare_vision_context_for_llm = Mock(return_value=None)
        cls.get_input = Mock()
        cls.terminal_get_input = Mock()
        cls.close_cv = Mock()
        cls.get_cv_state = Mock()

        rag_module = _stub_module(
            "rag.belt_v3_rag",
            rag_search=Mock(return_value=[]),
        )
        with patch.dict(sys.modules, {"rag.belt_v3_rag": rag_module}):
            sys.modules.pop("belt_v3_helper", None)
            helper_module = importlib.import_module("belt_v3_helper")

        stub_modules = {
            "movement.belt_v3_simple_action_handle": _stub_module(
                "movement.belt_v3_simple_action_handle",
                DEFAULT_COOLDOWN_SECONDS=5.0,
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
            "belt_v3_new_api": _stub_module(
                "belt_v3_new_api",
                ConversationMessage=dict,
                call_llm=cls.call_llm,
                remember_conversation_turn=cls.remember_conversation_turn,
            ),
            "belt_v3_helper": _stub_module(
                "belt_v3_helper",
                combine_spoken_parts=helper_module.combine_spoken_parts,
                compose_response=cls.compose_response,
                get_optional_cv_state=helper_module.get_optional_cv_state,
                prepare_vision_context_for_llm=(
                    cls.prepare_vision_context_for_llm
                ),
                print_timing=helper_module.print_timing,
                print_timing_summary=helper_module.print_timing_summary,
                speak_output=helper_module.speak_output,
            ),
            "belt_v3_input": _stub_module(
                "belt_v3_input",
                get_input=cls.get_input,
                terminal_get_input=cls.terminal_get_input,
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
        self.main_module.conversation.clear()

        for dependency in (
            self.simple_action_handle,
            self.speech_handle,
            self.testing_speech_handle,
            self.preload_qwen_model,
            self.navigation_handle,
            self.call_llm,
            self.remember_conversation_turn,
            self.compose_response,
            self.prepare_vision_context_for_llm,
            self.get_input,
            self.terminal_get_input,
            self.close_cv,
            self.get_cv_state,
        ):
            dependency.reset_mock()
            dependency.side_effect = None

        self.navigation_handle.return_value = ""

    def test_terminal_main_loop_processes_one_complete_response(self) -> None:
        response = {
            "output_list": [
                {
                    "type": "speech",
                    "text": "Hello!",
                },
                {
                    "type": "action",
                    "name": "wave",
                },
                {
                    "type": "speech",
                    "text": "Here are your directions.",
                },
                {
                    "type": "navigation",
                    "location": "2004",
                },
            ],
        }
        self.main_module.USING_ROBOT = False
        self.terminal_get_input.side_effect = [
            "Hi, Belt",
            KeyboardInterrupt,
        ]
        self.compose_response.return_value = (response, [])
        self.navigation_handle.return_value = (
            "To get to 2004, You have arrived!"
        )

        self.main_module.main()

        self.compose_response.assert_called_once()
        self.assertFalse(self.main_module.USE_DEEPSEEK_API)
        self.assertIs(
            self.compose_response.call_args.kwargs["llm_caller"],
            self.call_llm,
        )
        self.assertEqual(
            self.testing_speech_handle.call_args_list,
            [
                call(
                    "Hello!",
                    self.main_module.VOICE,
                ),
                call(
                    "Here are your directions. "
                    "To get to 2004, You have arrived!",
                    self.main_module.VOICE,
                ),
            ],
        )
        self.navigation_handle.assert_called_once_with("2004")
        self.simple_action_handle.assert_not_called()
        self.speech_handle.assert_not_called()
        self.preload_qwen_model.assert_not_called()
        self.get_input.assert_not_called()
        self.get_cv_state.assert_not_called()
        self.remember_conversation_turn.assert_called_once_with(
            self.main_module.conversation,
            "Hi, Belt",
            "Hello! Here are your directions. "
            "To get to 2004, You have arrived!",
        )
        self.close_cv.assert_not_called()

    def test_robot_main_loop_passes_disabled_wake_word_setting(self) -> None:
        response = {
            "output_list": [
                {
                    "type": "speech",
                    "text": "Hello!",
                },
            ],
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
        execution_order = []
        response = {
            "output_list": [
                {
                    "type": "speech",
                    "text": "I can help with that.",
                },
                {
                    "type": "action",
                    "name": "wave",
                },
                {
                    "type": "speech",
                    "text": "Here are your directions.",
                },
                {
                    "type": "navigation",
                    "location": "2110",
                },
            ],
        }
        timings = {"output_audio": 0.0}
        self.speech_handle.side_effect = (
            lambda text, _voice: execution_order.append(
                ("speech", text)
            )
        )
        self.simple_action_handle.side_effect = (
            lambda actions: execution_order.append(
                ("action", actions)
            )
        )
        self.navigation_handle.side_effect = (
            lambda location: (
                execution_order.append(
                    ("navigation", location)
                )
                or "To get to 2110,\nYou have arrived!"
            )
        )

        spoken_response = self.main_module.execute_modules(
            response,
            timings,
        )

        self.assertEqual(
            self.speech_handle.call_args_list,
            [
                call(
                    "I can help with that.",
                    self.main_module.VOICE,
                ),
                call(
                    "Here are your directions. "
                    "To get to 2110, You have arrived!",
                    self.main_module.VOICE,
                ),
            ],
        )
        self.simple_action_handle.assert_called_once_with(["wave"])
        self.navigation_handle.assert_called_once_with("2110")
        self.assertEqual(
            execution_order,
            [
                ("speech", "I can help with that."),
                ("action", ["wave"]),
                ("navigation", "2110"),
                (
                    "speech",
                    "Here are your directions. "
                    "To get to 2110, You have arrived!",
                ),
            ],
        )
        self.assertEqual(
            spoken_response,
            "I can help with that. Here are your directions. "
            "To get to 2110, You have arrived!",
        )
        self.assertGreaterEqual(timings["output_audio"], 0.0)

    def test_adjacent_actions_receive_a_cooldown(self) -> None:
        self.main_module.USING_ROBOT = True
        response = {
            "output_list": [
                {
                    "type": "speech",
                    "text": "Watch this.",
                },
                {
                    "type": "action",
                    "name": "wave",
                },
                {
                    "type": "action",
                    "name": "clap",
                },
                {
                    "type": "speech",
                    "text": "All done.",
                },
            ],
        }
        timings = {"output_audio": 0.0}

        with patch.object(self.main_module.time, "sleep") as sleep:
            spoken_response = self.main_module.execute_modules(
                response,
                timings,
            )

        self.assertEqual(
            self.simple_action_handle.call_args_list,
            [
                call(["wave"]),
                call(["clap"]),
            ],
        )
        sleep.assert_called_once_with(
            self.main_module.DEFAULT_COOLDOWN_SECONDS
        )
        self.assertEqual(spoken_response, "Watch this. All done.")

    def test_terminal_actions_are_simulated_without_robot_calls(self) -> None:
        self.main_module.USING_ROBOT = False
        response = {
            "output_list": [
                {
                    "type": "speech",
                    "text": "Watch this.",
                },
                {
                    "type": "action",
                    "name": "wave",
                },
                {
                    "type": "action",
                    "name": "clap",
                },
                {
                    "type": "speech",
                    "text": "All done.",
                },
            ],
        }
        timings = {"output_audio": 0.0}

        with patch.object(self.main_module.time, "sleep") as sleep:
            spoken_response = self.main_module.execute_modules(
                response,
                timings,
            )

        self.simple_action_handle.assert_not_called()
        self.speech_handle.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(
            self.testing_speech_handle.call_args_list,
            [
                call("Watch this.", self.main_module.VOICE),
                call("All done.", self.main_module.VOICE),
            ],
        )
        self.assertEqual(spoken_response, "Watch this. All done.")

    def test_response_pipeline_has_no_anonymous_functions(self) -> None:
        belt_v3_directory = Path(__file__).resolve().parent

        for filename in (
            "belt_v3_main.py",
            "belt_v3_helper.py",
            "belt_v3_api.py",
            "belt_v3_new_api.py",
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

    def test_main_defines_only_pipeline_functions(self) -> None:
        main_path = Path(__file__).resolve().parent / "belt_v3_main.py"
        syntax_tree = ast.parse(main_path.read_text(encoding="utf-8"))
        function_names = [
            node.name
            for node in syntax_tree.body
            if isinstance(node, ast.FunctionDef)
        ]

        self.assertEqual(
            function_names,
            ["generate_response", "execute_modules", "main"],
        )


if __name__ == "__main__":
    unittest.main()

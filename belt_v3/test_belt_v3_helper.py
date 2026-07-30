"""Tests for the ordered LLM output-list format."""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch


rag_search = Mock(return_value=[])
rag_module = types.ModuleType("rag.belt_v3_rag")
rag_module.rag_search = rag_search

with patch.dict(
    sys.modules,
    {"rag.belt_v3_rag": rag_module},
):
    sys.modules.pop("belt_v3_helper", None)
    helper = importlib.import_module("belt_v3_helper")


class OrderedOutputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        rag_search.reset_mock()
        rag_search.return_value = []

    def test_validates_order_and_removes_unsupported_actions(self) -> None:
        raw_response = {
            "output_list": [
                {"type": "speech", "text": " Hi! "},
                {"type": "action", "name": "WAVE"},
                {"type": "action", "name": "backflip"},
                {"type": "speech", "text": "How can I help?"},
                {"type": "navigation", "location": "break room"},
                {"type": "action", "name": "wave"},
            ],
        }

        self.assertEqual(
            helper._validated_llm_response(raw_response),
            {
                "output_list": [
                    {"type": "speech", "text": "Hi!"},
                    {"type": "action", "name": "wave"},
                    {
                        "type": "speech",
                        "text": "How can I help?",
                    },
                    {
                        "type": "navigation",
                        "location": "BREAK ROOM",
                    },
                    {"type": "action", "name": "wave"},
                ],
            },
        )

    def test_requires_new_format_and_at_least_one_speech_event(self) -> None:
        fallback = helper._fallback_response()

        self.assertEqual(
            helper._validated_llm_response(
                {
                    "speech": "Old format",
                    "simple_action": {
                        "requested": True,
                        "actions": ["wave"],
                    },
                }
            ),
            fallback,
        )
        self.assertEqual(
            helper._validated_llm_response(
                {
                    "output_list": [
                        {"type": "action", "name": "wave"},
                    ],
                }
            ),
            fallback,
        )

    def test_prompt_lists_valid_movements_and_ordered_schema(self) -> None:
        prompt = helper.build_response_prompt(
            "Wave and say hello",
            "No relevant document information found.",
        )

        self.assertIn('"output_list"', prompt)
        self.assertIn('"type": "speech"', prompt)
        self.assertIn('"type": "action"', prompt)
        self.assertIn('"type": "navigation"', prompt)
        self.assertIn("Never invent", prompt)
        for movement_name in helper.VALID_MOVEMENT_NAMES:
            self.assertIn(
                json.dumps(movement_name),
                prompt,
            )

    def test_compose_response_uses_selected_caller_and_new_format(
        self,
    ) -> None:
        llm_caller = Mock(
            return_value=json.dumps(
                {
                    "output_list": [
                        {
                            "type": "speech",
                            "text": "Hello.",
                        },
                        {
                            "type": "action",
                            "name": "wave",
                        },
                    ],
                }
            )
        )

        response, rag_context = helper.compose_response(
            "Please wave",
            [],
            llm_caller=llm_caller,
        )

        self.assertEqual(
            response,
            {
                "output_list": [
                    {
                        "type": "speech",
                        "text": "Hello.",
                    },
                    {
                        "type": "action",
                        "name": "wave",
                    },
                ],
            },
        )
        self.assertEqual(
            rag_context,
            "No relevant document information found.",
        )
        llm_caller.assert_called_once()
        self.assertEqual(
            llm_caller.call_args.kwargs["conversation"],
            [],
        )


if __name__ == "__main__":
    unittest.main()

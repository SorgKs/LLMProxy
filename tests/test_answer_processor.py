# test_answer_processor.py
import json
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answers import AnswerProcessor


class TestRealLLMResponse(unittest.TestCase):
    """Тест на реальном ответе LLM"""

    def setUp(self):
        self.processor = AnswerProcessor()
        self.processor.workspace_path = "/home/sorg/ZFS/Dev/LLMProxy"

    def create_answer_from_real_response(self, response_data: dict):
        """Создает Answer из реального ответа LLM"""
        answer = Mock()
        answer.model = response_data.get("model", "unknown")
        answer.duration = response_data.get("duration_seconds", 1.0)
        answer.is_stream = False
        answer.status_code = response_data.get("status_code", 200)
        answer.full_response = response_data.get("response", {})

        choices = answer.full_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            answer.content = message.get("content", "")
            answer.tool_calls = message.get("tool_calls", [])
        else:
            answer.content = ""
            answer.tool_calls = []

        return answer

    def test_real_response_3e3b(self):
        """Тест на реальном ответе"""
        real_response = {
            "timestamp": "2026-05-05T00:33:28.462542",
            "status_code": 200,
            "duration_seconds": 9.149383783340454,
            "response": {
                "id": "gen-1777930399-9AlD6ykKsqaAEjr8jGal",
                "object": "chat.completion",
                "created": 1777930399,
                "model": "inclusionai/ling-2.6-1t-20260423:free",
                "provider": "Novita",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "index": 0,
                                    "id": "fc-b336f734-e044-46a9-a982-07cee265703a",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\": \"requests.py\"}"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        answer = self.create_answer_from_real_response(real_response)
        self.processor.process(answer)

        self.assertTrue(len(answer.tool_calls) > 0)
        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)
        self.assertEqual(parsed['path'], "requests.py")

        print("✅ test_real_response_3e3b passed!")

    def test_real_response_3e3b_arguments_structure(self):
        """Проверяет конкретную структуру arguments после process()"""
        real_response = {
            "response": {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "type": "function",
                            "index": 0,
                            "id": "fc-test",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"requests.py\"}"
                            }
                        }]
                    }
                }]
            }
        }

        answer = self.create_answer_from_real_response(real_response)
        self.processor.process(answer)

        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)

        self.assertIsInstance(parsed, dict)
        self.assertIn('path', parsed)
        self.assertEqual(parsed['path'], "requests.py")
        self.assertIn('mode', parsed)
        self.assertEqual(parsed['mode'], 'slice')

        print(f"✅ parsed = {parsed}")

    def test_processor_adds_mode_slice(self):
        """Проверяет, что процессор добавляет mode='slice' к read_file"""
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"test.py\"}"
            }
        }]

        answer = self.create_answer_from_real_response(
            {"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}}
        )
        self.processor.process(answer)

        args_after = answer.tool_calls[0]['function']['arguments']
        parsed_after = json.loads(args_after)

        self.assertIn('path', parsed_after)
        self.assertEqual(parsed_after['path'], "test.py")
        self.assertIn('mode', parsed_after)
        self.assertEqual(parsed_after['mode'], 'slice')

        print(f"✅ After: {parsed_after}")

    def test_processor_fixes_double_encoding(self):
        """Проверяет, что процессор исправляет двойное экранирование"""
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"requests.py\"}"
            }
        }]

        answer = self.create_answer_from_real_response(
            {"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}}
        )
        self.processor.process(answer)

        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)
        self.assertEqual(parsed['path'], "requests.py")
        self.assertEqual(parsed.get('mode'), 'slice')

        print(f"✅ Fixed args: {parsed}")

    def test_processor_search_files_double_encoding(self):
        """Проверяет, что процессор обрабатывает search_files без ошибок"""
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "search_files",
                "arguments": "{\"path\": \".\", \"regex\": \"RequestProcessor\"}"
            }
        }]

        answer = self.create_answer_from_real_response(
            {"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}}
        )

        try:
            self.processor.process(answer)
        except Exception as e:
            self.fail(f"process() raised an exception: {e}")

        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)
        self.assertEqual(parsed['path'], ".")
        self.assertIn('regex', parsed)

        print(f"✅ search_files parsed: {parsed}")

    def test_processor_list_files_double_encoding(self):
        """Проверяет, что процессор обрабатывает list_files без ошибок"""
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "list_files",
                "arguments": "{\"path\": \".\", \"recursive\": true}"
            }
        }]

        answer = self.create_answer_from_real_response(
            {"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}}
        )

        try:
            self.processor.process(answer)
        except Exception as e:
            self.fail(f"process() raised an exception: {e}")

        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)
        self.assertEqual(parsed['path'], ".")
        self.assertTrue(parsed['recursive'])

        print(f"✅ list_files parsed: {parsed}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

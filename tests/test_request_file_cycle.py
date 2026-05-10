# tests/test_request_file_cycle.py
"""
Интеграционные тесты полного цикла request_file:
1. proxy.py получает запрос, вызывает answers.process()
2. process() возвращает action='request_file' (нужен контент файла)
3. proxy.py отвечает клиенту с read_file tool_call
4. Клиент присылает read_file response в новом запросе
5. proxy.py извлекает данные через _extract_file_content_from_request
6. proxy.py вызывает process() с данными → получаем apply_diff
7. proxy.py отвечает клиенту с apply_diff tool_call
"""
import os
import sys
import json
import asyncio
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx


class TestRequestFileCycle(unittest.TestCase):
    """Тесты полного цикла request_file через ASGI transport."""

    @classmethod
    def setUpClass(cls):
        """Загружаем тестовые случаи из JSON-файла."""
        cls.test_file = os.path.join(
            os.path.dirname(__file__), 'assets', 'test_request_file_cycle.json'
        )
        with open(cls.test_file, 'r', encoding='utf-8') as f:
            cls.test_cases = json.load(f)

    def _mock_send_to_llm(self, mock_llm_response):
        """Патч send_to_llm_with_retry для возврата mock-ответа."""
        async def _mock_send(*args, **kwargs):
            return mock_llm_response
        return _mock_send

    def _mock_process(self, mock_process_result):
        """Патч answer_processor.process для возврата нужного результата."""
        def _mock_process(answer, request_body=None, data=None):
            return mock_process_result
        return _mock_process

    async def _test_case(self, case):
        """Выполняет один тестовый кейс через ASGI."""
        import proxy as proxy_module
        import importlib
        importlib.reload(proxy_module)

        request = case["request"]
        mock_llm = case.get("mock_llm")
        mock_process = case.get("mock_process")
        expected = case["expected"]

        transport = httpx.ASGITransport(app=proxy_module.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            patches = []

            if mock_llm:
                patches.append(
                    patch("proxy.send_to_llm_with_retry",
                          side_effect=self._mock_send_to_llm(mock_llm))
                )

            if mock_process is not None:
                patches.append(
                    patch.object(proxy_module.answer_processor, "process",
                                 side_effect=self._mock_process(mock_process))
                )

            for p in patches:
                p.start()

            try:
                response = await client.post(
                    "/v1/chat/completions",
                    json=request,
                )
            finally:
                for p in reversed(patches):
                    p.stop()

            self.assertEqual(
                response.status_code, expected["status_code"],
                f"Status code mismatch: expected {expected['status_code']}, got {response.status_code}"
            )

            result = response.json()
            choices = result.get("choices", [])

            if "tool_call_name" in expected:
                self.assertTrue(len(choices) > 0, f"No choices in response")
                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls", [])

                if expected["tool_call_name"] is None:
                    self.assertEqual(
                        len(tool_calls), 0,
                        f"Expected no tool_calls, got {tool_calls}"
                    )
                else:
                    self.assertTrue(
                        len(tool_calls) > 0,
                        f"Expected tool_call '{expected['tool_call_name']}', got none"
                    )
                    actual_name = tool_calls[0].get("function", {}).get("name")
                    self.assertEqual(
                        actual_name, expected["tool_call_name"],
                        f"Tool call name mismatch: expected '{expected['tool_call_name']}', got '{actual_name}'"
                    )

            if "content_contains" in expected:
                content = choices[0].get("message", {}).get("content", "")
                self.assertIn(
                    expected["content_contains"], content,
                    f"Content doesn't contain '{expected['content_contains']}'"
                )

    def test_all_json_cases(self):
        """Проверяет все кейсы из JSON-файла через ASGI."""
        if not self.test_cases:
            self.skipTest("Файл с тестовыми случаями пуст или не найден")

        for case in self.test_cases:
            name = case.get("name", "Без имени")
            with self.subTest(name=name):
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._test_case(case))
                finally:
                    loop.close()
                print(f"  ✅ case '{name}' passed")

        print(f"\n✅ Проверено {len(self.test_cases)} интеграционных кейсов request_file cycle")

    def test_extract_file_content_from_request_in_proxy(self):
        """Тестирует _extract_file_content_from_request напрямую через proxy."""
        from proxy import _extract_file_content_from_request

        body = {
            "messages": [
                {
                    "role": "tool",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({
                                    "path": "src/main.py",
                                    "content": "import os\n\ndef hello():\n    print('hello')\n",
                                    "line_count": 4,
                                    "EOF": True,
                                    "offset": 1
                                })
                            }
                        }
                    ]
                }
            ]
        }

        result = _extract_file_content_from_request(body, "src/main.py")

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "file_content")
        self.assertEqual(result["path"], "src/main.py")
        self.assertTrue(result["EOF"])
        self.assertEqual(result["line_count"], 4)
        self.assertEqual(result["content"]["1"], "import os")
        self.assertEqual(result["content"]["2"], "")
        self.assertEqual(result["content"]["3"], "def hello():")
        self.assertEqual(result["content"]["4"], "    print('hello')")

    async def _test_full_cycle_async(self):
        """
        Полный цикл через ASGI:
        1. POST с function_replace → process() возвращает request_file
        2. Ответ содержит read_file tool_call
        3. POST с read_file response → process() возвращает replace_with_apply_diff
        4. Ответ содержит apply_diff tool_call
        """
        import proxy as proxy_module
        import importlib
        importlib.reload(proxy_module)

        transport = httpx.ASGITransport(app=proxy_module.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:

            request_body = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Change add to multiply"}
                ],
                "tools": [
                    {"type": "function", "function": {"name": "function_replace"}}
                ]
            }

            # Шаг 1: LLM возвращает function_replace
            mock_llm_response = {
                "full_response": {
                    "id": "test-100",
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "tc-req-1",
                                "type": "function",
                                "function": {
                                    "name": "function_replace",
                                    "arguments": json.dumps({
                                        "path": "calc.py",
                                        "function": "add",
                                        "full_code": "def add(a, b):\n    return a * b\n"
                                    })
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                },
                "duration": 0.1,
                "status_code": 200,
                "is_error": False,
                "is_stream": False,
                "headers": {}
            }

            request_file_action = {
                "action": "request_file",
                "path": "calc.py",
                "function_name": "add",
                "full_code": "def add(a, b):\n    return a * b\n",
                "original_tc": {
                    "id": "tc-req-1",
                    "type": "function",
                    "function": {
                        "name": "function_replace",
                        "arguments": json.dumps({
                            "path": "calc.py",
                            "function": "add",
                            "full_code": "def add(a, b):\n    return a * b\n"
                        })
                    }
                }
            }

            with patch(
                "proxy.send_to_llm_with_retry",
                side_effect=self._mock_send_to_llm(mock_llm_response),
            ), patch.object(
                proxy_module.answer_processor,
                "process",
                side_effect=self._mock_process(request_file_action),
            ):
                response = await client.post("/v1/chat/completions", json=request_body)

            self.assertEqual(response.status_code, 200)
            result = response.json()
            choices = result["choices"]
            self.assertEqual(len(choices), 1)
            tool_calls = choices[0]["message"]["tool_calls"]
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0]["function"]["name"], "read_file")
            read_file_args = json.loads(tool_calls[0]["function"]["arguments"])
            self.assertEqual(read_file_args["path"], "calc.py")

            print("  ✅ Step 1: function_replace → read_file passed")

            # Шаг 2: Клиент присылает read_file response с данными файла
            apply_diff_action = {
                "action": "replace_with_apply_diff",
                "path": "calc.py",
                "diff": "--- calc.py\\n+++ calc.py\\n@@ -1,2 +1,2 @@\\n"
                        " def add(a, b):\\n"
                        "-    return a + b\\n"
                        "+    return a * b\\n",
                "tool_call": {
                    "id": "tc-apply-1",
                    "type": "function",
                    "function": {
                        "name": "apply_diff",
                        "arguments": json.dumps({
                            "path": "calc.py",
                            "diff": "--- calc.py\\n+++ calc.py\\n@@ -1,2 +1,2 @@\\n"
                                    " def add(a, b):\\n"
                                    "-    return a + b\\n"
                                    "+    return a * b\\n"
                        })
                    }
                }
            }

            # Устанавливаем _process_request вручную чтобы симулировать состояние после шага 1
            proxy_module.answer_processor._process_request["calc.py"] = {
                "function_name": "add",
                "full_code": "def add(a, b):\n    return a * b\n",
                "request_body": request_body.copy()
            }

            with patch(
                "proxy.send_to_llm_with_retry",
                side_effect=self._mock_send_to_llm(mock_llm_response),
            ), patch.object(
                proxy_module.answer_processor,
                "process",
                side_effect=self._mock_process(apply_diff_action),
            ):
                response2 = await client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [
                        {"role": "user", "content": "Change add to multiply"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "tc-req-1",
                                "type": "function",
                                "function": {
                                    "name": "function_replace",
                                    "arguments": json.dumps({
                                        "path": "calc.py",
                                        "function": "add",
                                        "full_code": "def add(a, b):\n    return a * b\n"
                                    })
                                }
                            }]
                        },
                        {
                            "role": "tool",
                            "tool_calls": [{
                                "id": "call_req_file_123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({
                                        "path": "calc.py",
                                        "content": "# Calculator\ndef add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
                                        "line_count": 5,
                                        "EOF": True,
                                        "offset": 1
                                    })
                                }
                            }]
                        }
                    ],
                    "tools": [
                        {"type": "function", "function": {"name": "function_replace"}}
                    ]
                })

            self.assertEqual(response2.status_code, 200)
            result2 = response2.json()
            choices2 = result2["choices"]
            self.assertEqual(len(choices2), 1)
            tool_calls2 = choices2[0]["message"]["tool_calls"]
            self.assertEqual(len(tool_calls2), 1)
            self.assertEqual(tool_calls2[0]["function"]["name"], "apply_diff")

            print("  ✅ Step 2: read_file response → apply_diff passed")

    def test_full_cycle_with_mocked_process_and_llm(self):
        """
        Полный цикл через ASGI:
        1. POST с function_replace → process() возвращает request_file
        2. Ответ содержит read_file tool_call
        3. POST с read_file response → process() возвращает replace_with_apply_diff
        4. Ответ содержит apply_diff tool_call
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._test_full_cycle_async())
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
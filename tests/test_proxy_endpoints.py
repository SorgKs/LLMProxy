# -*- coding: utf-8 -*-
"""
Тесты для proxy.py (API endpoints)
Проверяет эндпоинты /v1/chat/completions, /v1/models, логику обработки ответов
"""
import os
import sys
import json
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnswerDataclass(unittest.TestCase):
    """Тесты для дата-класса Answer"""

    def test_answer_creation(self):
        """Проверяет создание объекта Answer"""
        from proxy import Answer

        answer = Answer(
            full_response={"model": "test-model", "choices": [{"message": {"content": "test"}}]},
            is_stream=False,
            status_code=200,
            duration=0.0
        )
        self.assertIsNotNone(answer)
        self.assertEqual(answer.model, "test-model")
        self.assertEqual(answer.content, "test")
        self.assertEqual(answer.tool_calls, [])
        self.assertFalse(answer.is_stream)
        self.assertEqual(answer.status_code, 200)

        print("✅ test_answer_creation passed!")

    def test_answer_content_property(self):
        """Проверяет setter для content"""
        from proxy import Answer

        answer = Answer(
            full_response={"choices": [{"message": {"content": ""}}]},
            is_stream=False,
            status_code=200,
            duration=0.0
        )
        answer.content = "Test content"
        self.assertEqual(answer.content, "Test content")

        print("✅ test_answer_content_property passed!")

    def test_answer_tool_calls_property(self):
        """Проверяет setter для tool_calls"""
        from proxy import Answer

        answer = Answer(
            full_response={"choices": [{"message": {}}]},
            is_stream=False,
            status_code=200,
            duration=0.0
        )
        calls = [{"type": "function", "function": {"name": "test"}}]
        answer.tool_calls = calls
        self.assertEqual(answer.tool_calls, calls)

        print("✅ test_answer_tool_calls_property passed!")


class TestExtractFileContentFromRequest(unittest.TestCase):
    """Тесты для функции _extract_file_content_from_request"""

    def test_extract_with_read_file(self):
        """Проверяет извлечение контента из read_file вызовов"""
        from proxy import _extract_file_content_from_request

        body = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "test.py", "content": "line1\nline2"}),
                            }
                        }
                    ]
                }
            ]
        }

        result = _extract_file_content_from_request(body)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "file_content")
        self.assertEqual(result["path"], "test.py")

        print("✅ test_extract_with_read_file passed!")

    def test_extract_no_tool_calls(self):
        """Проверяет извлечение при отсутствии tool_calls"""
        from proxy import _extract_file_content_from_request

        body = {"messages": []}

        result = _extract_file_content_from_request(body)
        self.assertIsNone(result)

        print("✅ test_extract_no_tool_calls passed!")


class TestCheckFileSufficiency(unittest.TestCase):
    """Тесты для функции check_file_sufficiency"""

    def test_sufficient_data(self):
        """Проверяет достаточные данные"""
        from proxy import check_file_sufficiency

        data = {
            "type": "file_content",
            "path": "test.py",
            "content": {"1": "line1", "2": "line2"},
            "EOF": True,
        }
        pending_info = {
            "path": "test.py",
            "function_name": "func",
            "full_code": "code",
        }

        result = check_file_sufficiency(data, pending_info)
        self.assertTrue(result)

        print("✅ test_sufficient_data passed!")

    def test_insufficient_data_no_eof(self):
        """Проверяет недостаточные данные (нет EOF)"""
        from proxy import check_file_sufficiency

        data = {
            "type": "file_content",
            "content": {"1": "line1"},
            "EOF": False,
        }
        pending_info = {
            "path": "test.py",
            "function_name": "func",
            "full_code": "code",
        }

        result = check_file_sufficiency(data, pending_info)
        self.assertFalse(result)

        print("✅ test_insufficient_data_no_eof passed!")


class TestSaveResponse(unittest.TestCase):
    """Тесты для функции save_responce"""

    def test_save_response_creates_file(self):
        """Проверяет создание файла ответа"""
        from proxy import save_responce

        test_response = {
            "id": "test-123",
            "choices": [{"message": {"content": "test"}}],
        }

        # Мокаем open, чтобы не создавать реальные файлы
        with patch("builtins.open", MagicMock()) as mock_open:
            with patch("os.makedirs"):
                save_responce(False, test_response)

        print("✅ test_save_response_creates_file passed!")

    def test_save_modified_response(self):
        """Проверяет сохранение модифицированного ответа"""
        from proxy import save_responce

        test_response = {
            "id": "test-456",
            "choices": [{"message": {"content": "modified"}}],
        }

        with patch("builtins.open", MagicMock()):
            with patch("os.makedirs"):
                save_responce(True, test_response)

        print("✅ test_save_modified_response passed!")


class TestCreateErrorResponse(unittest.TestCase):
    """Тесты для функции create_error_response"""

    def test_create_error_response_structure(self):
        """Проверяет структуру ответа с ошибкой"""
        from proxy import create_error_response
        import json

        collected_data = {
            "error": "Bad request",
            "status_code": 400,
        }

        response = create_error_response(
            collected_data=collected_data,
            is_stream=False,
        )

        self.assertEqual(response.status_code, 400)
        response_body = json.loads(response.body)
        self.assertIn("error", response_body)
        self.assertEqual(response_body["error"]["message"], "Bad request")
        self.assertEqual(response_body["error"]["type"], "api_error")
        self.assertEqual(response_body["error"]["code"], 400)

        print("✅ test_create_error_response_structure passed!")

    def test_create_error_response_defaults(self):
        """Проверяет значения по умолчанию"""
        from proxy import create_error_response

        collected_data = {
            "error": "Error",
            "status_code": 500,
        }

        response = create_error_response(
            collected_data=collected_data,
            is_stream=False,
        )

        self.assertEqual(response.status_code, 500)

        print("✅ test_create_error_response_defaults passed!")


class TestSendToLLMWithRetry(unittest.TestCase):
    """Тесты для функции send_to_llm_with_retry"""

    def test_send_to_llm_returns_response(self):
        """Проверяет, что функция возвращает ответ"""
        from proxy import send_to_llm_with_retry

        # Мокаем httpx.AsyncClient
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.aread = AsyncMock(return_value=b'{"choices": []}')
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            # Запускаем асинхронно
            result = asyncio.run(
                send_to_llm_with_retry(
                    request_body={"model": "test"},
                    headers={},
                )
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["status_code"], 200)

        print("✅ test_send_to_llm_returns_response passed!")


class TestProxyInitialization(unittest.TestCase):
    """Тесты инициализации proxy"""

    def test_proxy_has_app(self):
        """Проверяет наличие FastAPI app"""
        # Мокаем модули, которые импортирует proxy
        with patch.dict('sys.modules', {'answers': MagicMock(), 'requests': MagicMock()}):
            from proxy import app

        self.assertIsNotNone(app)
        self.assertTrue(hasattr(app, "routes"))

        print("✅ test_proxy_has_app passed!")

    def test_proxy_routes_exist(self):
        """Проверяет наличие основных роутов"""
        # Мокаем модули, которые импортирует proxy
        with patch.dict('sys.modules', {'answers': MagicMock(), 'requests': MagicMock()}):
            from proxy import app

        routes = [route.path for route in app.routes]
        self.assertIn("/v1/chat/completions", routes)
        self.assertTrue(
            any("/v1/models" in route for route in routes)
            or any("/models" in route for route in routes)
        )

        print("✅ test_proxy_routes_exist passed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

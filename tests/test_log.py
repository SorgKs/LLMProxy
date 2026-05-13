# -*- coding: utf-8 -*-
import log
"""
Тесты для log.py (логирование)
Проверяет функции структурированного логирования в JSON
"""
import os
import sys
import json
import unittest
import tempfile
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import log


class TestLogInfo(unittest.TestCase):
    """Тесты для log_info"""

    def test_log_info_outputs_json(self):
        """Проверяет, что log_info записывает валидный JSON"""
        with patch("log._write_log") as mock_write_log:
            log.log_info("Test message", key="value")

            # Проверяем, что _write_log был вызван
            self.assertTrue(mock_write_log.called)

            # Получаем аргументы вызова
            call_args = mock_write_log.call_args
            data = call_args[0][0]

            # Проверяем структуру
            self.assertIn("message", data)
            self.assertEqual(data["message"], "Test message")
            self.assertIn("key", data)
            self.assertEqual(data["key"], "value")
            self.assertEqual(data["level"], "INFO")

        log.log_info("test_log_info_outputs_json passed!")

    def test_log_info_with_timestamp(self):
        """Проверяет наличие временной метки"""
        with patch("log._write_log") as mock_write_log:
            log.log_info("Test")

            data = mock_write_log.call_args[0][0]

            self.assertIn("timestamp", data)

        log.log_info("test_log_info_with_timestamp passed!")


class TestLogDebug(unittest.TestCase):
    """Тесты для log_debug"""

    def test_log_debug_outputs_json(self):
        """Проверяет, что log_debug записывает валидный JSON"""
        with patch("log._write_log") as mock_write_log:
            log.log_debug("Debug message", data="test")

            data = mock_write_log.call_args[0][0]

            self.assertEqual(data["message"], "Debug message")
            self.assertEqual(data["level"], "DEBUG")
            self.assertEqual(data["data"], "test")

        log.log_info("test_log_debug_outputs_json passed!")


class TestLogRequest(unittest.TestCase):
    """Тесты для log_request"""

    def test_log_request_structure(self):
        """Проверяет структуру лога запроса"""
        with patch("log._write_log") as mock_write_log:
            log.log_request("POST", "http://test.com", {"Authorization": "Bearer xxx"})

            data = mock_write_log.call_args[0][0]

            self.assertEqual(data["type"], "request")
            self.assertEqual(data["method"], "POST")
            self.assertEqual(data["url"], "http://test.com")
            self.assertIn("headers", data)

        log.log_info("test_log_request_structure passed!")

    def test_log_request_masks_headers(self):
        """Проверяет маскировку заголовков авторизации"""
        with patch("log._write_log") as mock_write_log:
            # Используем lowercase "authorization", так как log_request проверяет именно его
            log.log_request(
                "POST", "http://test.com", {"authorization": "Bearer secret-token-12345"}
            )

            data = mock_write_log.call_args[0][0]

            # Токен должен быть замаскирован
            auth_header = data["headers"].get("authorization", "")
            self.assertEqual(auth_header, "***")

        log.log_info("test_log_request_masks_headers passed!")


class TestLogModifiedRequest(unittest.TestCase):
    """Тесты для log_modified_request"""

    def test_log_modified_request_structure(self):
        """Проверяет структуру лога модифицированного запроса"""
        with patch("log._write_log") as mock_write_log:
            log.log_modified_request(
                method="POST",
                url="http://test.com",
                headers={"Content-Type": "application/json"},
                modifications=["modification1", "modification2"],
            )

            data = mock_write_log.call_args[0][0]

            self.assertEqual(data["type"], "modified_request")
            self.assertIn("modifications", data)
            self.assertEqual(len(data["modifications"]), 2)

        log.log_info("test_log_modified_request_structure passed!")


class TestLogResponse(unittest.TestCase):
    """Тесты для log_response"""

    def test_log_response_structure(self):
        """Проверяет структуру лога ответа"""
        with patch("log._write_log") as mock_write_log:
            log.log_response(1.5, status_code=200)

            data = mock_write_log.call_args[0][0]

            self.assertEqual(data["type"], "response")
            self.assertEqual(data["duration"], 1.5)
            self.assertEqual(data["status_code"], 200)

        log.log_info("test_log_response_structure passed!")

    def test_log_response_default_status(self):
        """Проверяет значение status_code по умолчанию"""
        with patch("log._write_log") as mock_write_log:
            log.log_response(2.0)

            data = mock_write_log.call_args[0][0]

            # По умолчанию должно быть 200
            self.assertEqual(data["status_code"], 200)

        log.log_info("test_log_response_default_status passed!")


class TestLogRetryAttempt(unittest.TestCase):
    """Тесты для log_retry_attempt"""

    def test_log_retry_attempt_structure(self):
        """Проверяет структуру лога попытки повтора"""
        with patch("log._write_log") as mock_write_log:
            log.log_retry_attempt(
                conversation_id="conv-123",
                tool_call_id="tc-456",
                attempt=1,
                errors=[{"error": "Invalid JSON"}],
                retry_message={"title": "Retry", "message": "Fix", "advice": "Check", "requires_attention": True},
            )

            data = mock_write_log.call_args[0][0]

            self.assertEqual(data["type"], "retry_attempt")
            self.assertEqual(data["conversation_id"], "conv-123")
            self.assertEqual(data["tool_call_id"], "tc-456")
            self.assertEqual(data["attempt"], 1)
            self.assertIn("errors", data)
            self.assertIn("retry_message", data)

        log.log_info("test_log_retry_attempt_structure passed!")


class TestLogStats(unittest.TestCase):
    """Тесты для log_stats"""

    def test_log_stats_structure(self):
        """Проверяет структуру лога статистики"""
        with patch("log._write_log") as mock_write_log:
            log.log_stats(
                model="gpt-4",
                duration=3.5,
                status_code=200,
                response_type="tool_calls",
            )

            data = mock_write_log.call_args[0][0]

            self.assertEqual(data["type"], "stats")
            self.assertEqual(data["level"], "DEBUG")
            self.assertEqual(data["model"], "gpt-4")
            self.assertEqual(data["duration"], 3.5)
            self.assertEqual(data["status_code"], 200)

        log.log_info("test_log_stats_structure passed!")


class TestWriteLog(unittest.TestCase):
    """Тесты для внутренней функции _write_log"""

    def test_write_log_to_file(self):
        """Проверяет запись лога в файл"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_file = f.name

        # Мокаем MAIN_LOG
        original_log_file = log.MAIN_LOG
        log.MAIN_LOG = log_file

        try:
            test_data = {"type": "test", "message": "Test log entry"}

            log._write_log(test_data)

            # Проверяем, что файл создан и содержит данные
            with open(log_file, "r") as f:
                content = f.read()
                self.assertIn("Test log entry", content)

        finally:
            log.MAIN_LOG = original_log_file
            if os.path.exists(log_file):
                os.unlink(log_file)

        log.log_info("test_write_log_to_file passed!")

    def test_write_log_handles_exception(self):
        """Проверяет обработку исключений при записи"""
        # Мокаем open, чтобы вызвать исключение
        with patch("builtins.open", side_effect=Exception("File error")):
            # Не должно вызывать исключение (ловится внутри)
            try:
                log._write_log({"type": "test"})
            except Exception:
                self.fail("_write_log должен обрабатывать исключения внутри")

        log.log_info("test_write_log_handles_exception passed!")


class TestLogIntegration(unittest.TestCase):
    """Интеграционные тесты логирования"""

    def test_multiple_log_calls(self):
        """Проверяет несколько последовательных вызовов логирования"""
        with patch("log._write_log") as mock_write_log:
            log.log_info("Message 1")
            log.log_info("Message 2")
            log.log_response(1.0)

            # Должно быть 3 вызова _write_log
            self.assertEqual(mock_write_log.call_count, 3)

        log.log_info("test_multiple_log_calls passed!")

    def test_log_with_special_characters(self):
        """Проверяет логирование со специальными символами"""
        with patch("log._write_log") as mock_write_log:
            special_message = "Test with special chars: \"quotes\", \\backslashes\\, \nnewlines"
            log.log_info(special_message)

            data = mock_write_log.call_args[0][0]

            self.assertEqual(data["message"], special_message)

        log.log_info("test_log_with_special_characters passed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

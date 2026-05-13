# -*- coding: utf-8 -*-
"""
Тесты для RequestProcessor (requests.py)
Проверяет обработку запросов, сообщений, системных промптов и инструментов
"""
import os
import sys
import tempfile
import yaml
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import log
from requests import RequestProcessor


class TestRequestProcessorInit(unittest.TestCase):
    """Тесты инициализации RequestProcessor"""

    def test_default_init(self):
        """Проверяет инициализацию с параметрами по умолчанию"""
        processor = RequestProcessor()
        self.assertIsNone(processor.workspace_path)
        self.assertEqual(processor.config_path, "config/tools.yaml")
        self.assertIsInstance(processor.tools_config, dict)
        self.assertIsInstance(processor.changes_log, list)
        log.log_info("test_default_init passed!")

    def test_custom_config_path(self):
        """Проверяет инициализацию с кастомным путем к конфигу"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"tools": {"read_file": True}}, f)
            config_path = f.name

        try:
            processor = RequestProcessor(config_path=config_path)
            self.assertEqual(processor.config_path, config_path)
            log.log_info("test_custom_config_path passed!")
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


class TestMessageProcessing(unittest.TestCase):
    """Тесты обработки сообщений"""

    def setUp(self):
        self.processor = RequestProcessor()

    def test_remove_markers_from_text(self):
        """Проверяет удаление маркеров из текста"""
        test_cases = [
            ("Hello <marker>world</marker>", "Hello <marker>world</marker>"),
            ("<marker>Remove me</marker> keep this", "<marker>Remove me</marker> keep this"),
            ("No markers here", "No markers here"),
        ]

        for input_text, expected in test_cases:
            result = self.processor._remove_markers(input_text)
            self.assertEqual(result, expected, f"Failed for: {input_text}")

        log.log_info("test_remove_markers_from_text passed!")

    def test_remove_system_prompt_noise(self):
        """Проверяет удаление шума из системного промпта"""
        result = self.processor._remove_system_prompt_noise("  Test \n\n\n noise  ")
        expected = "Test \n\n noise"
        self.assertEqual(result, expected)

        log.log_info("test_remove_system_prompt_noise passed!")

    def test_optimize_system_prompt(self):
        """Проверяет оптимизацию системного промпта"""
        input_text = "System prompt without special sections"
        result = self.processor._optimize_system_prompt(input_text)
        self.assertEqual(result, input_text)

        log.log_info("test_optimize_system_prompt passed!")

    def test_process_messages_with_system_prompt(self):
        """Проверяет обработку сообщений с системным промптом"""
        messages = [
            {"role": "system", "content": "You are a helper \n\n\n noise  "},
            {"role": "user", "content": "Hello"},
        ]

        changed, original_total, new_total = self.processor._process_messages(messages)

        self.assertTrue(changed)
        self.assertIsInstance(original_total, int)
        self.assertIsInstance(new_total, int)
        self.assertEqual(messages[0]["content"], "You are a helper \n\n noise")

        log.log_info("test_process_messages_with_system_prompt passed!")

    def test_process_messages_no_system_prompt(self):
        """Проверяет обработку сообщений без системного промпта"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        changed, original_total, new_total = self.processor._process_messages(messages)

        self.assertFalse(changed)
        self.assertEqual(messages[0]["content"], "Hello")

        log.log_info("test_process_messages_no_system_prompt passed!")


class TestToolProcessing(unittest.TestCase):
    """Тесты обработки инструментов"""

    def setUp(self):
        self.processor = RequestProcessor()

    def test_process_tools_removes_descriptions(self):
        """Проверяет оптимизацию описаний инструментов"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Very long description that should be optimized or removed",
                },
            },
        ]

        changed, filtered_tools = self.processor._process_tools(tools)

        self.assertTrue(changed)
        self.assertIsInstance(filtered_tools, list)
        self.assertNotEqual(
            filtered_tools[0]["function"]["description"],
            "Very long description that should be optimized or removed"
        )

        log.log_info("test_process_tools_removes_descriptions passed!")

    def test_process_tools_read_file_optimization(self):
        """Проверяет специфическую оптимизацию для read_file"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads a file from the filesystem",
                },
            },
        ]

        changed, filtered_tools = self.processor._process_tools(tools)

        self.assertTrue(changed)
        self.assertEqual(filtered_tools[0]["function"]["name"], "read_file")
        expected_desc = "Read a file from the local filesystem. returns up to 2000 lines per file"
        self.assertEqual(filtered_tools[0]["function"]["description"], expected_desc)

        log.log_info("test_process_tools_read_file_optimization passed!")

    def test_process_tools_empty_list(self):
        """Проверяет обработку пустого списка инструментов"""
        changed, filtered_tools = self.processor._process_tools([])
        self.assertFalse(changed)
        self.assertEqual(filtered_tools, [])

        log.log_info("test_process_tools_empty_list passed!")

    def test_process_tools_none(self):
        """Проверяет обработку None вместо списка инструментов"""
        with self.assertRaises(TypeError):
            self.processor._process_tools(None)

        log.log_info("test_process_tools_none passed!")


class TestProjectStructure(unittest.TestCase):
    """Тесты добавления структуры проекта"""

    def test_get_project_structure(self):
        """Проверяет получение структуры проекта"""
        processor = RequestProcessor(workspace_path=".")

        structure = processor._get_project_structure()

        if structure is not None:
            self.assertIsInstance(structure, str)
            self.assertGreater(len(structure), 0)

        log.log_info("test_get_project_structure passed!")

    def test_add_project_structure_to_request(self):
        """Проверяет добавление структуры проекта в запрос"""
        processor = RequestProcessor(workspace_path=".")

        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = processor._add_project_structure_to_request(body)

        self.assertIsInstance(result, bool)

        log.log_info("test_add_project_structure_to_request passed!")


class TestFullProcess(unittest.TestCase):
    """Тесты полной обработки запроса"""

    def test_process_request_with_messages(self):
        """Проверяет полную обработку запроса с сообщениями"""
        processor = RequestProcessor()

        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helper \n\n\n noise  "},
                {"role": "user", "content": "Process this"},
            ],
        }

        result = processor.process(body, method="POST", url="/v1/chat/completions")

        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertIsInstance(result["messages"], list)
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["content"], "You are a helper \n\n noise")

        log.log_info("test_process_request_with_messages passed!")

    def test_process_request_with_tools(self):
        """Проверяет полную обработку запроса с инструментами"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"tools": {"read_file": True, "apply_diff": True}}, f)
            config_path = f.name

        try:
            processor = RequestProcessor(config_path=config_path)

            body = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read file",
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "apply_diff",
                            "description": "Apply diff",
                        },
                    },
                ],
            }

            result = processor.process(body)

            self.assertIn("tools", result)
            self.assertTrue(len(result["tools"]) > 0)

            log.log_info("test_process_request_with_tools passed!")
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_process_request_no_modifications(self):
        """Проверяет запрос без необходимости модификаций"""
        processor = RequestProcessor()

        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Simple message"}],
        }

        result = processor.process(body)

        self.assertEqual(result["messages"][0]["content"], "Simple message")

        log.log_info("test_process_request_no_modifications passed!")


class TestReset(unittest.TestCase):
    """Тесты сброса состояния процессора"""

    def test_reset_clears_changes_log(self):
        """Проверяет, что reset очищает лог изменений"""
        processor = RequestProcessor()
        messages = [{"role": "system", "content": "test \n\n\n noise"}]
        processor._process_messages(messages)

        self.assertGreater(len(processor.changes_log), 0)

        processor.reset()

        self.assertEqual(len(processor.changes_log), 0)

        log.log_info("test_reset_clears_changes_log passed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

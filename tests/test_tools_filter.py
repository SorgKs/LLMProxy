# -*- coding: utf-8 -*-
"""
Тест фильтрации инструментов (tools) по конфигурации config/tools.yaml
"""
import os
import tempfile
import yaml
import json

from requests import RequestProcessor


def test_filter_tools_by_config_true_kept_false_removed():
    """
    Проверяет, что инструменты со значением True в конфигурации остаются,
    а со значением False — фильтруются (удаляются).
    """
    # Создаём временный YAML-файл с mixed true/false значениями
    config_content = {
        "tools": {
            "apply_diff": True,
            "read_file": False,
            "ask_followup_question": True,
            "attempt_completion": False,
            "codebase_search": True,
            "skill": False,
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_content, f)
        config_file = f.name

    try:
        processor = RequestProcessor(config_path=config_file)

        # Проверяем загруженную конфигурацию
        assert processor.tools_config["apply_diff"] is True
        assert processor.tools_config["read_file"] is False
        assert processor.tools_config["ask_followup_question"] is True
        assert processor.tools_config["attempt_completion"] is False
        assert processor.tools_config["codebase_search"] is True
        assert processor.tools_config["skill"] is False

        # Подготавливаем список инструментов (в формате OpenAI tools)
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "apply_diff",
                    "description": "Применяет изменения к файлу",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Считывает файл",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_followup_question",
                    "description": "Задаёт уточняющий вопрос",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attempt_completion",
                    "description": "Попытка завершения",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "codebase_search",
                    "description": "Ищет в коде",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "skill",
                    "description": "Специальный навык",
                },
            },
        ]

        # Фильтруем через метод процессора
        filtered = processor._filter_tools_by_config(test_tools)

        # Ожидаем только те, у которых True в конфиге
        expected_names = {"apply_diff", "ask_followup_question", "codebase_search"}
        filtered_names = {t["function"]["name"] for t in filtered}

        assert filtered_names == expected_names, (
            f"Ожидались {expected_names}, получены {filtered_names}"
        )

        # Убеждаемся, что удалённых инструментов в списке нет
        removed_names = {"read_file", "attempt_completion", "skill"}
        assert removed_names.isdisjoint(filtered_names), (
            f"Удалённые инструменты остались в списке: {removed_names & filtered_names}"
        )

        # Проверяем, что в лог изменений добавилась запись об удалении
        assert any(
            "удалены" in msg.lower() or "removed" in msg.lower()
            for msg in processor.changes_log
        ), "В changes_log должна быть запись об удалении инструментов"

        print("✅ test_filter_tools_by_config_true_kept_false_removed passed!")

    finally:
        # Убираем временный файл
        if os.path.exists(config_file):
            os.unlink(config_file)


def test_process_request_body_filters_tools():
    """
    Проверяет, что при вызове process() с телом запроса,
    содержащим tools, инструменты с false удаляются из итогового body.
    """
    config_content = {
        "tools": {
            "apply_diff": True,
            "read_file": False,
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_content, f)
        config_file = f.name

    try:
        processor = RequestProcessor(config_path=config_file)

        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Привет"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "apply_diff",
                        "description": "Применяет изменения",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Считывает файл",
                    },
                },
            ],
        }

        result = processor.process(body, method="POST", url="/v1/chat/completions")

        # В результате должен остаться только apply_diff
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "apply_diff"

        print("✅ test_process_request_body_filters_tools passed!")

    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)


def test_empty_tools_list():
    """
    Проверяет, что при пустом списке tools фильтрация ничего не ломает.
    """
    config_content = {
        "tools": {
            "apply_diff": True,
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_content, f)
        config_file = f.name

    try:
        processor = RequestProcessor(config_path=config_file)

        filtered = processor._filter_tools_by_config([])
        assert filtered == []

        # Также проверяем None
        filtered2 = processor._filter_tools_by_config(None)
        assert filtered2 is None or filtered2 == []

        print("✅ test_empty_tools_list passed!")

    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)


if __name__ == "__main__":
    test_filter_tools_by_config_true_kept_false_removed()
    test_process_request_body_filters_tools()
    test_empty_tools_list()
    print("\\n🎉 Все тесты пройдены!")

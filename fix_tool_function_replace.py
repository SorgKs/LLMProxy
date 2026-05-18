# fix_tool_function_replace.py
"""
Класс для конвертации function_replace tool calls в apply_diff формат.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FixToolFunctionReplace:
    """
    Класс для конвертации function_replace в apply_diff формат.
    """

    def __init__(self):
        self._was_changed = False

    @property
    def changed(self) -> bool:
        """Были ли изменения"""
        return self._was_changed

    def fix(
        self,
        path: str,
        function_name: str,
        full_code: str,
        content_dict: Dict,
        tool_call: dict,
    ) -> bool:
        """
        Полная обработка function_replace: конвертация в apply_diff и обновление tool_call.

        Args:
            path: Путь к файлу
            function_name: Имя функции
            full_code: Новый код функции
            content_dict: Словарь {номер_строки: строка} с содержимым файла
            tool_call: Tool call для модификации (меняется на месте)

        Returns:
            True если конвертация успешна, иначе False
        """
        if not content_dict:
            logger.warning("   ❌ content_dict пуст")
            return False

        logger.debug(f"   ✓ Вызов _convert...")
        diff_result = self._convert(path, function_name, full_code, content_dict)

        if diff_result:
            logger.debug(f"   ✓ Конвертация успешна!")
            tool_call['function']['name'] = 'apply_diff'
            tool_call['function']['arguments'] = {'path': path, 'diff': diff_result['diff']}
            return True
        else:
            logger.warning(f"   ❌ convert вернул None")
            return False

    def convert(
        self,
        path: str,
        function_name: str,
        full_code: str,
        file_lines: Dict[int, str],
    ) -> Optional[Dict[str, str]]:
        """
        Конвертирует function_replace в apply_diff формат.

        Args:
            path: Путь к файлу
            function_name: Имя функции
            full_code: Новый код функции
            file_lines: Словарь {номер_строки: строка} с содержимым файла

        Returns:
            Dict с path и diff или None при ошибке
        """
        if not file_lines:
            logger.error(f"function_replace: словарь file_lines пуст")
            return None

        if not isinstance(file_lines, dict):
            logger.error(f"function_replace: file_lines должен быть dict, получен {type(file_lines)}")
            return None

        # Проверяем, что ключи - int
        if not all(isinstance(k, int) for k in file_lines.keys()):
            logger.error(f"function_replace: ключи file_lines должны быть int, получены {type(next(iter(file_lines.keys())))}")
            return None

        # Собираем старый код из словаря в правильном порядке
        try:
            sorted_keys = sorted(file_lines.keys())
            old_code_lines = [file_lines[k] for k in sorted_keys]
            old_code = '\n'.join(old_code_lines)
        except Exception as e:
            logger.error(f"function_replace: ошибка при сборке кода: {e}")
            return None

        if not old_code.strip():
            logger.error(f"function_replace: нет содержимого функции '{function_name}' в {path}")
            return None

        # Формируем diff
        diff_lines = [
            "<<<<<<< SEARCH",
            ":start_line:1",
            "-------",
            old_code,
            "=======",
            full_code,
            ">>>>>>> REPLACE"
        ]

        self._was_changed = True
        return {"path": path, "diff": '\n'.join(diff_lines)}

    def reset(self):
        """Сброс состояния"""
        self._was_changed = False
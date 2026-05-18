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
        tc: dict,
        data: dict,
        debug: bool = False,
    ) -> bool:
        """
        Полная обработка function_replace: конвертация в apply_diff и обновление tc.

        Args:
            tc: Tool call для модификации (меняется на месте)
            data: Данные файла {"type": "function_content", "content": {...}}
            debug: Режим отладки

        Returns:
            True если конвертация успешна, иначе False
        """
        if not data or data.get("type") != "function_content":
            logger.warning("   ❌ Нет данных function_content")
            return False

        content_dict = data.get("content", {})
        if not content_dict:
            logger.warning("   ❌ content_dict пуст")
            return False

        # Извлекаем параметры из tc
        func_args = tc.get('function', {}).get('arguments', {})
        path = func_args.get('path', '')
        function_name = func_args.get('function', '')
        full_code = func_args.get('full_code', '')

        if debug:
            logger.debug(f"   path: {path}")
            logger.debug(f"   function_name: {function_name}")
            logger.debug(f"   full_code length: {len(full_code)}")

        logger.debug(f"   ✓ Вызов _convert...")
        result = self._convert(path, function_name, full_code, content_dict)

        if result:
            logger.debug(f"   ✓ Конвертация успешна!")
            tc['function']['name'] = 'apply_diff'
            tc['function']['arguments'] = {'path': path, 'diff': result['diff']}
            self._was_changed = True
            return True
        else:
            logger.warning(f"   ❌ _convert вернул None")
            return False

    def _convert(
        self,
        path: str,
        function_name: str,
        full_code: str,
        content_dict: Dict[int, str],
    ) -> Optional[Dict[str, str]]:
        """
        Конвертирует function_replace в apply_diff формат.

        Args:
            path: Путь к файлу
            function_name: Имя функции
            full_code: Новый код функции
            content_dict: Словарь {номер_строки: строка} с содержимым файла

        Returns:
            Dict с path и diff или None при ошибке
        """
        if not content_dict:
            logger.error(f"function_replace: словарь content_dict пуст")
            return None

        if not isinstance(content_dict, dict):
            logger.error(f"function_replace: content_dict должен быть dict, получен {type(content_dict)}")
            return None

        # Проверяем, что ключи - int
        if not all(isinstance(k, int) for k in content_dict.keys()):
            logger.error(f"function_replace: ключи content_dict должны быть int, получены {type(next(iter(content_dict.keys())))}")
            return None

        # Собираем старый код из словаря в правильном порядке
        try:
            sorted_keys = sorted(content_dict.keys())
            old_code_lines = [content_dict[k] for k in sorted_keys]
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
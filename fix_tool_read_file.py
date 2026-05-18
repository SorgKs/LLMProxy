# fix_tool_read_file.py
"""
Класс для исправления и валидации read_file tool calls.
"""
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FixToolReadFile:
    """
    Класс для комплексного исправления read_file:
    - добавление mode=slice
    - исправление offset с 0 на 1
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
        debug: bool = False,
    ) -> Tuple[bool, list]:
        """
        Комплексное исправление read_file.

        Args:
            tc: Tool call для модификации (меняется на месте)
            debug: Режим отладки

        Returns:
            Tuple[bool, list]: (были ли исправления, список изменений)
        """
        changed = False
        changes = []

        parsed_args = tc.get('function', {}).get('arguments', {})

        if not isinstance(parsed_args, dict):
            logger.warning("FixToolFixReadFile.fix: arguments не dict")
            return False, changes

        # Добавляем mode=slice если отсутствует
        if self._add_mode_slice(parsed_args):
            changed = True
            changes.append("read_file: добавлен mode=slice")

        # Исправляем offset с 0 на 1
        if self._fix_offset(parsed_args):
            changed = True
            changes.append("read_file: offset исправлен с 0 на 1")

        if changed:
            self._was_changed = True

        return changed, changes

    def _add_mode_slice(self, parsed_args: dict) -> bool:
        """
        Добавляет mode=slice в read_file если отсутствует.

        Args:
            parsed_args: Dict с аргументами (уже распарсенными)

        Returns:
            True если были изменения, иначе False
        """
        if "mode" not in parsed_args:
            parsed_args["mode"] = "slice"
            return True
        return False

    def _fix_offset(self, parsed_args: dict) -> bool:
        """
        Исправляет offset с 0 на 1 в read_file.

        Args:
            parsed_args: Dict с аргументами (уже распарсенными)

        Returns:
            True если были изменения, иначе False
        """
        if "offset" in parsed_args and parsed_args["offset"] == 0:
            parsed_args["offset"] = 1
            return True
        return False

    def reset(self):
        """Сброс состояния"""
        self._was_changed = False

# fix_tool_ask_followup_question.py
"""
Класс для исправления и валидации ask_followup_question tool calls.
"""
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FixToolAskFollowupQuestion:
    """
    Класс для комплексного исправления ask_followup_question:
    - конвертация follow_up из строки в массив
    - нормализация массива строк в массив объектов
    """

    def __init__(self):
        self._was_changed = False

    @property
    def changed(self) -> bool:
        """Были ли изменения"""
        return self._was_changed

    def fix(
        self,
        parsed_args: dict,
        debug: bool = False,
    ) -> bool:
        """
        Комплексное исправление ask_followup_question.

        Args:
            parsed_args: Dict с аргументами (уже распарсенными)
            debug: Режим отладки

        Returns:
            True если были изменения, иначе False
        """
        changed = False

        if "follow_up" not in parsed_args:
            return False

        follow_up = parsed_args["follow_up"]

        # Case 1: это строка с JSON
        if isinstance(follow_up, str):
            try:
                parsed = json.loads(follow_up)
                if isinstance(parsed, list):
                    parsed_args["follow_up"] = self._normalize_follow_up_array(parsed)
                    changed = True
            except json.JSONDecodeError:
                pass

        # Case 2: это массив строк
        elif isinstance(follow_up, list) and follow_up and isinstance(follow_up[0], str):
            parsed_args["follow_up"] = self._normalize_follow_up_array(follow_up)
            changed = True

        if changed:
            self._was_changed = True

        return changed

    def _normalize_follow_up_array(self, arr: list) -> list:
        """
        Преобразует массив строк в массив объектов {text: "...", mode: null}

        Args:
            arr: Массив строк

        Returns:
            Массив объектов {text: "...", mode: null}
        """
        return [{"text": item, "mode": None} for item in arr]

    def reset(self):
        """Сброс состояния"""
        self._was_changed = False

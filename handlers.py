# handlers.py
"""Обработчики запросов и валидация данных"""

import time
import json


def validate_fields(func_name: str, obj_name: str, data, fields: list) -> None:
    """
    Проверяет что все обязательные поля присутствуют в словаре или объекте.
    Выбрасывает ValueError если поля отсутствуют.
    
    Args:
        func_name: имя функции для сообщения об ошибке
        obj_name: имя объекта для сообщения об ошибке
        data: словарь или объект для проверки
        fields: список обязательных полей/атрибутов
        
    Raises:
        ValueError: Если отсутствуют обязательные поля/атрибуты
    """
    if data is None:
        raise ValueError(f"В {func_name} {obj_name} is None")
    
    # Определяем тип и проверяем наличие полей
    missed_fields = []
    
    if isinstance(data, dict):
        # Для словаря проверяем ключи
        missed_fields = [f for f in fields if f not in data]
        field_type = "ключи"
    else:
        # Для объекта проверяем атрибуты
        missed_fields = [f for f in fields if not hasattr(data, f)]
        field_type = "атрибуты"
    
    if missed_fields:
        raise ValueError(
            f"В {func_name} {obj_name} не содержит обязательные {field_type} {missed_fields}"
        )
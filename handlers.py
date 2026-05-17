# handlers.py
"""Обработчики запросов и валидация данных"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class Colors:
    """ANSI цвета для вывода"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ArgumentParseError(Exception):
    """Исключение при ошибке парсинга аргументов tool call"""
    def __init__(self, message: str, tool_name: str = None, tool_call_id: str = None, original_args: str = None):
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.original_args = original_args
        super().__init__(message)


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


# ============================================================================
# UNIVERSAL SAVE AND CLEANUP FUNCTIONS
# ============================================================================

def cleanup_folder(folder_path: str, max_age_hours: int = 24, max_files: int = 30) -> None:
    """
    Универсальная функция для очистки папки: хранит только за последние N часов,
    но не более M файлов.
    
    Args:
        folder_path: Путь к папке для очистки
        max_age_hours: Максимальный возраст файлов в часах (по умолчанию 24)
        max_files: Максимальное количество файлов (по умолчанию 30)
    """
    try:
        if not os.path.exists(folder_path):
            return
        
        # Получаем все json файлы
        files = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                stat = os.stat(filepath)
                files.append({
                    'path': filepath,
                    'mtime': stat.st_mtime
                })
        
        if not files:
            return
        
        # Сортируем по времени (новейшие первые)
        files.sort(key=lambda x: x['mtime'], reverse=True)
        
        now = datetime.now()
        cutoff = now.timestamp() - max_age_hours * 60 * 60
        
        # Файлы для удаления: старше max_age_hours или превышают лимит max_files
        files_to_delete = []
        for i, f in enumerate(files):
            if f['mtime'] < cutoff:
                files_to_delete.append(f['path'])
            elif i >= max_files:
                files_to_delete.append(f['path'])
        
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
            except Exception:
                pass  # Игнорируем ошибки удаления отдельных файлов
                
    except Exception:
        pass  # Игнорируем ошибки при очистке


def save_json_content(
    folder: str,
    content: Dict[str, Any],
    prefix: str,
    cleanup: bool = True,
    cleanup_max_age_hours: int = 24,
    cleanup_max_files: int = 30
) -> Optional[str]:
    """
    Универсальная функция для сохранения JSON контента в файл.
    
    Args:
        folder: Папка для сохранения (будет создана если не существует)
        content: Словарь с данными для сохранения
        prefix: Префикс имени файла (например, "original_request" или "modified_request")
        cleanup: Выполнять очистку папки после сохранения
        cleanup_max_age_hours: Максимальный возраст файлов в часах для очистки
        cleanup_max_files: Максимальное количество файлов для очистки
        
    Returns:
        Путь к сохраненному файлу или None при ошибке
    """
    try:
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}/{prefix}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        
        # Очищаем старые файлы после сохранения
        if cleanup:
            cleanup_folder(folder, cleanup_max_age_hours, cleanup_max_files)
        
        return filename
    except Exception:
        return None
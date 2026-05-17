# answers.py
import json
import logging
import re
import time
import copy
import os
import yaml
import handlers
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from fix_tool_apply_diff import FixToolApplyDiff

logger = logging.getLogger(__name__)


class AnswerProcessor:
    """
    Класс для обработки и исправления tool calls от LLM.
    process() ИЗМЕНЯЕТ исходные данные напрямую.
    Свойство changed показывает, были ли изменения.
    """
    
    def __init__(self, config_path: str = "config/tools.yaml"):
        """Инициализация процессора"""
        self._was_changed = False
        self.changes_log = []
        self._current_test_name = ""  # Для тестов
        self.progress = {}
        self._fix_tool_apply_diff = FixToolApplyDiff()
        
        # Загружаем конфигурацию инструментов
        self.tools_config = self._load_tools_config(config_path)
        
        # Загружаем шаблоны сообщений для retry
        self.retry_messages = self._load_retry_messages()
        
        # Сигнатуры для поиска маркеров
        self.source_signatures = [
            r'^\[?SEARCH\]?$',
            r'^\[?SOURCE\]?$',
            r'^\[?SRC\]?$',
            r'^<<<<<<< SEARCH$',
            r'^<<<<<<< SOURCE$',
            r'^<<<<<<< SRC$',
        ]
        
        self.replace_signatures = [
            r'^\[?REPLACE\]?$',
            r'^>>>>>>> REPLACE$',
        ]
        
        self.common_signatures = [
            r'^={3,}$',
            r'^-{3,}$',
            r'^>{3,}$',
        ]
    
    def _load_retry_messages(self) -> Dict:
        """
        Загружает шаблоны сообщений для retry из config/retry_messages.yaml
        
        Returns:
            Dict с шаблонами сообщений
        """
        config_path = "config/retry_messages.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config if isinstance(config, dict) else {}
    
    @property
    def changed(self) -> bool:
        """Были ли изменения в последнем process()"""
        return self._was_changed
    
    def _load_tools_config(self, config_path: str) -> Dict[str, bool]:
        """
        Загружает конфигурацию инструментов из config/tools.yaml
        
        Args:
            config_path: Путь к файлу конфигурации
            
        Returns:
            Dict[str, bool]: Словарь {имя_инструмента: включен/выключен}
        """

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tools_config = {}
        tools = config.get("tools")
        if tools and isinstance(tools, dict):
            for tool_name, tool_enabled in tools.items():
                tools_config[tool_name] = bool(tool_enabled)
        
        return tools_config
    
    def _parse_arguments(self, args_str: str, tool_name: str = "unknown", tool_call_id: str = "unknown") -> Dict[str, Any]:
        """Парсит строку аргументов в dict. Выбрасывает handlers.ArgumentParseError при ошибке."""
        if isinstance(args_str, dict):
            return args_str
        
        if not isinstance(args_str, str):
            raise handlers.ArgumentParseError(
                f"Неверный тип аргументов: ожидается str или dict, получен {type(args_str)}",
                tool_name=tool_name, tool_call_id=tool_call_id, original_args=str(args_str)
            )
        
        if not args_str.strip():
            raise handlers.ArgumentParseError(
                "Пустая строка аргументов",
                tool_name=tool_name, tool_call_id=tool_call_id, original_args=args_str
            )
        
        # 0. Раскрываем двойное экранирование
        args_str = self._unwrap_double_encoding(args_str)

        # 1. Пробуем как JSON
        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, dict):
                return parsed
            raise handlers.ArgumentParseError(
                f"JSON распарсен, но результат не dict (тип: {type(parsed).__name__})",
                tool_name=tool_name, tool_call_id=tool_call_id, original_args=args_str
            )
        except json.JSONDecodeError:
            pass
        
        # 2. Пробуем распарсить Python-подобный синтаксис
        try:
            return self._args_to_dict(args_str, tool_name, tool_call_id)
        except handlers.ArgumentParseError:
            pass
        
        raise handlers.ArgumentParseError(
            "Не удалось распарсить аргументы ни одним из способов",
            tool_name=tool_name, tool_call_id=tool_call_id, original_args=args_str[:500]
        )
    
    def _serialize_arguments(self, args_dict: Dict[str, Any]) -> str:
        """Сериализует dict в JSON строку"""
        if not args_dict:
            return "{}"
        
        serializable = copy.deepcopy(args_dict)
        
        if "diff" in serializable and isinstance(serializable["diff"], str):
            serializable["diff"] = serializable["diff"].replace('\\\\n', '\\n')
        
        return json.dumps(serializable, ensure_ascii=False)
    
    def _convert_function_replace(
        self,
        path: str,
        function_name: str,
        full_code: str,
        file_lines: Dict[int, str],
    ) -> Optional[Dict[str, str]]:
        
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
        
        return {"path": path, "diff": '\n'.join(diff_lines)}

    def validate_tool_calls_exist(
        self,
        tool_calls: List[Dict],
        available_tools: List[str]
    ) -> Tuple[bool, List[Dict]]:
        """
        Проверяет, существуют ли вызываемые инструменты.
        Учитывает множественные определения инструментов в конфигурации.
        """
        invalid = []
        
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            tool_call_id = tc.get("id", "unknown")
            
            # Проверяем, есть ли инструмент в available_tools
            if tool_name not in available_tools:
                tool_config = self.tools_config.get(tool_name, True)
                invalid.append({
                    "tool_call": tc,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "available_tools": available_tools,
                    "config_value": tool_config,
                    "reason": f"tool '{tool_name}' not in available_tools list"
                })
        
        # ВЫВОД ТОЛЬКО ПРИ ОШИБКАХ
        if invalid:
            logger.warning(f"НЕСУЩЕСТВУЮЩИЕ ИНСТРУМЕНТЫ:")
            logger.warning(f"   Всего: {len(invalid)}")
            for inv in invalid:
                logger.warning(f"   - {inv['tool_name']} (ID: {inv['tool_call_id']})")
                logger.warning(f"     Причина: {inv['reason']}")
                logger.warning(f"     Доступные инструменты: {inv['available_tools']}")
        
        return len(invalid) == 0, invalid

    def reset(self):
        """Сброс состояния"""
        self._was_changed = False
        self.changes_log = []
        self.progress = {}

    def _decode_unicode_escapes(self, text: str) -> str:
        """
        Декодирует Unicode escape последовательности в тексте.
        Например: \u0417\u043d\u0430\u0435\u0442\u0435 -> Знаете
        
        Args:
            text: Текст с возможными Unicode escape последовательностями
            
        Returns:
            Текст с декодированными Unicode символами
        """
        if not isinstance(text, str) or '\\u' not in text:
            return text
        
        try:
            return text.encode('utf-8').decode('unicode_escape')
        except:
            return text
    
    def _decode_all_strings(self, obj: Any) -> Any:
        """
        Рекурсивно декодирует все строки в объекте.
        
        Args:
            obj: Любой объект (dict, list, str, и т.д.)
            
        Returns:
            Объект с декодированными строками
        """
        if isinstance(obj, dict):
            return {k: self._decode_all_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._decode_all_strings(item) for item in obj]
        elif isinstance(obj, str):
            return self._decode_unicode_escapes(obj)
        else:
            return obj

    def generate_retry_message(self, validation_result: Dict, content_issues: List[Dict], 
                              formatting_issues: List[Dict], attempt_count: int = 0) -> Dict:
        """
        Генерирует структурированное сообщение для retry на основе ошибок валидации.
        
        Args:
            validation_result: Результат validate_apply_diff
            content_issues: Список проблем с содержимым
            formatting_issues: Список проблем с форматированием
            attempt_count: Текущее количество попыток
            
        Returns:
            Dict с сообщением для retry
        """
        
        # Если нет ошибок - возвращаем пустое сообщение
        if not content_issues and not formatting_issues:
            return {
                "type": "success",
                "message": "Валидация пройдена успешно"
            }
        
        # Сортируем ошибки по приоритету
        all_issues = []
        
        for issue in content_issues:
            issue_type = issue["type"]
            template = self.retry_messages["content"].get(issue_type, {
                "title": "❌ Неизвестная ошибка",
                "message": issue["message"],
                "advice": "Пожалуйста, проверьте запрос и повторите.",
                "priority": 99
            })
            all_issues.append({
                **issue,
                "template": template,
                "category": "content"
            })
        
        for issue in formatting_issues:
            issue_type = issue["type"]
            template = self.retry_messages["formatting"].get(issue_type, {
                "title": "⚠️ Неизвестная ошибка форматирования",
                "message": issue["message"],
                "advice": "Пожалуйста, проверьте формат diff.",
                "internal": True,
                "priority": 99
            })
            all_issues.append({
                **issue,
                "template": template,
                "category": "formatting"
            })
        
        # Сортируем по приоритету (меньше = важнее)
        all_issues.sort(key=lambda x: x["template"].get("priority", 99))
        
        # Формируем основное сообщение
        if len(content_issues) == 1 and not formatting_issues:
            # Одна content ошибка
            issue = content_issues[0]
            template = self.retry_messages["content"][issue["type"]]
            result = {
                "type": "retry_notice",
                "title": template["title"],
                "message": template["message"],
                "details": [issue["message"]],
                "advice": template["advice"],
                "requires_attention": True,
                "internal_notes": []
            }
        
        elif len(formatting_issues) == 1 and not content_issues:
            # Одна formatting ошибка
            issue = formatting_issues[0]
            template = self.retry_messages["formatting"][issue["type"]]
            result = {
                "type": "retry_notice",
                "title": template["title"],
                "message": template["message"],
                "details": [issue["message"]],
                "advice": template["advice"],
                "requires_attention": False,
                "internal_notes": ["⚠️ Баг в алгоритме исправления форматирования"]
            }
        
        elif content_issues and not formatting_issues:
            # Только content ошибки, несколько
            template = self.retry_messages["combined"]["multiple_content_issues"]
            details = []
            for issue in content_issues[:3]:  # Показываем первые 3
                details.append(f"• {issue['message']}")
            if len(content_issues) > 3:
                details.append(f"• ...и еще {len(content_issues) - 3} проблем")
            
            result = {
                "type": "retry_notice",
                "title": template["title"].format(count=len(content_issues)),
                "message": template["message"].format(count=len(content_issues)),
                "details": details,
                "advice": template["advice"],
                "requires_attention": True,
                "internal_notes": []
            }
        
        elif formatting_issues and not content_issues:
            # Только formatting ошибки, несколько
            template = self.retry_messages["combined"]["multiple_formatting_issues"]
            details = []
            for issue in formatting_issues[:3]:
                details.append(f"• {issue['message']}")
            if len(formatting_issues) > 3:
                details.append(f"• ...и еще {len(formatting_issues) - 3} ошибок")
            
            result = {
                "type": "retry_notice",
                "title": template["title"].format(count=len(formatting_issues)),
                "message": template["message"].format(count=len(formatting_issues)),
                "details": details,
                "advice": template["advice"],
                "requires_attention": False,
                "internal_notes": ["⚠️ Множественные баги в алгоритме исправления форматирования"]
            }
        
        else:
            # Смешанные ошибки
            template = self.retry_messages["combined"]["mixed_issues"]
            details = []
            
            # Добавляем content issues
            if content_issues:
                details.append("📁 Проблемы с содержимым:")
                for issue in content_issues[:2]:
                    details.append(f"  • {issue['message']}")
                if len(content_issues) > 2:
                    details.append(f"  • ...и еще {len(content_issues) - 2}")
            
            # Добавляем formatting issues
            if formatting_issues:
                details.append("⚙️ Ошибки форматирования:")
                for issue in formatting_issues[:2]:
                    details.append(f"  • {issue['message']}")
                if len(formatting_issues) > 2:
                    details.append(f"  • ...и еще {len(formatting_issues) - 2}")
            
            result = {
                "type": "retry_notice",
                "title": template["title"],
                "message": template["message"].format(
                    content_count=len(content_issues),
                    formatting_count=len(formatting_issues)
                ),
                "details": details,
                "advice": template["advice"],
                "requires_attention": True,
                "internal_notes": ["⚠️ Обнаружены ошибки форматирования"] if formatting_issues else []
            }
        
        # Добавляем информацию о попытке
        result["attempt_info"] = {
            "current": attempt_count + 1,
            "max": 2,
            "remaining": 2 - (attempt_count + 1)
        }
        
        return result

    def _unwrap_double_encoding(self, args_str: str, max_depth: int = 10) -> str:
        """
        Рекурсивно парсит JSON до тех пор, пока это возможно.
        Возвращает строковое представление последнего успешно распарсенного объекта.
        """
        if not isinstance(args_str, str):
            # Если уже не строка - возвращаем как есть
            return args_str if isinstance(args_str, str) else json.dumps(args_str)
        
        current = args_str
        last_success = args_str
        last_success_obj = None
        
        for depth in range(max_depth):
            try:
                # Парсим текущий уровень
                parsed = json.loads(current)
                
                # Запоминаем успешный результат
                last_success = current
                last_success_obj = parsed
                
                # Если распарсили в строку - пробуем парсить её как JSON
                if isinstance(parsed, str):
                    current = parsed  # Продолжаем с этой строкой
                    continue
                
                # Если распарсили в dict или list - это конечный объект
                # Сериализуем обратно в строку (но уже без лишних кавычек)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(parsed)
                
                # Другие типы (числа, булевы, null) - возвращаем как есть
                return current
                
            except json.JSONDecodeError:
                # Невалидный JSON - выходим из цикла
                break
        
        # Возвращаем последний успешно распарсенный результат
        if last_success_obj is not None:
            if isinstance(last_success_obj, (dict, list)):
                return json.dumps(last_success_obj)
            elif isinstance(last_success_obj, str):
                # Может быть, нужно ещё раскрыть unicode escape?
                # Но в рамках этой логики - возвращаем как есть
                return last_success_obj
            else:
                return json.dumps(last_success_obj)
        
        # Ни одного успешного парсинга - возвращаем оригинал
        return args_str

    def _load_tools_format_config(self) -> Dict:
        """Загружает конфиг форматов инструментов"""
        config_path = "config/tools_format.yaml"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config.get('tools', {})
        except Exception as e:
            logger.error(f"Ошибка загрузки {config_path}: {e}")
        return {}

    def _check_field_type(self, value: Any, expected_type: str) -> bool:
        """Проверяет тип поля"""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        expected = type_map.get(expected_type)
        if not expected:
            return True
        return isinstance(value, expected)

    def validate_tool_calls(self, answer) -> Tuple[bool, List[Dict]]:
        """Валидирует все tool calls в ответе на основе конфига"""
        errors = []
        
        # Загружаем конфиг форматов (один раз)
        if not hasattr(self, '_format_config'):
            self._format_config = self._load_tools_format_config()
        
        for idx, tc in enumerate(answer.tool_calls):
            if 'function' not in tc:
                errors.append({
                    "tool_call_id": tc.get('id', 'unknown'),
                    "index": idx,
                    "error": "missing_function",
                    "message": "Tool call missing 'function' field"
                })
                continue
            
            func = tc['function']
            tool_name = func.get('name', 'unknown')
            tool_call_id = tc.get('id', 'unknown')
            
            # Получаем формат инструмента из конфига
            tool_format = self._format_config.get(tool_name, {})
            required_fields = [f['name'] for f in tool_format.get('fields', []) if f.get('required', False)]
            
            # Парсим аргументы
            args = func.get('arguments')
            
            # 1. arguments должен быть строкой
            if not isinstance(args, str):
                errors.append({
                    "tool_call_id": tool_call_id,
                    "index": idx,
                    "tool_name": tool_name,
                    "error": "invalid_type",
                    "message": f"Arguments must be string, got {type(args).__name__}"
                })
                continue
            
            # 2. arguments должен быть валидным JSON
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError as e:
                errors.append({
                    "tool_call_id": tool_call_id,
                    "index": idx,
                    "tool_name": tool_name,
                    "error": "invalid_json",
                    "message": f"Arguments is not valid JSON: {e}"
                })
                continue
            
            # 3. Распарсенный результат должен быть объектом
            if not isinstance(parsed, dict):
                errors.append({
                    "tool_call_id": tool_call_id,
                    "index": idx,
                    "tool_name": tool_name,
                    "error": "not_an_object",
                    "message": f"Arguments must parse to object, got {type(parsed).__name__}"
                })
                continue
            
            # 4. Проверяем обязательные поля из конфига
            if required_fields:
                missing_fields = [f for f in required_fields if f not in parsed]
                if missing_fields:
                    errors.append({
                        "tool_call_id": tool_call_id,
                        "index": idx,
                        "tool_name": tool_name,
                        "error": "missing_required_field",
                        "message": f"Required field(s) missing: {', '.join(missing_fields)}",
                        "required_fields": required_fields,
                        "missing_fields": missing_fields,
                        "received_fields": list(parsed.keys())
                    })
            
            # 5. Проверяем типы полей (опционально)
            for field_config in tool_format.get('fields', []):
                field_name = field_config['name']
                if field_name in parsed:
                    expected_type = field_config.get('type')
                    if expected_type:
                        actual_value = parsed[field_name]
                        if not self._check_field_type(actual_value, expected_type):
                            errors.append({
                                "tool_call_id": tool_call_id,
                                "index": idx,
                                "tool_name": tool_name,
                                "error": "invalid_field_type",
                                "message": f"Field '{field_name}' must be {expected_type}, got {type(actual_value).__name__}",
                                "field": field_name,
                                "expected_type": expected_type,
                                "received_type": type(actual_value).__name__
                            })
        
        return len(errors) == 0, errors

    def _fix_apply_diff_deep_search(self, diff: str, file_content: Dict[int, str], debug: bool = False) -> Tuple[Optional[str], List[str]]:
        """
        Глубокое восстановление diff при множественных разделителях.
        
        Args:
            diff: Исходный проблемный diff
            file_content: Содержимое файла {номер_строки: строка}
            debug: Режим отладки
            
        Returns:
            Tuple[Optional[str], List[str]]: (исправленный_diff, список_проблем)
        """
        issues = []
        lines = diff.split('\n')
        
        if debug:
            print(f"\n{handlers.Colors.CYAN}{'='*60}{handlers.Colors.RESET}")
            print(f"{handlers.Colors.BOLD}Deep Search: Анализ проблемного diff{handlers.Colors.RESET}")
            print(f"{handlers.Colors.CYAN}{'='*60}{handlers.Colors.RESET}")
            self._print_diff_preview(lines, "Исходный diff")
        
        # Фаза 1: Анализ маркеров в diff и извлечение контента
        remaining_lines = lines.copy()
        
        # Паттерны для поиска
        start_patterns = [
            r'^[\[<]*\s*(?:SEARCH|SOURCE|SRC)\s*[\>\] ]*$',
            r'^[<]+$',
            r'^[=]+$',
            r'^[-]+$',
        ]
        
        separator_patterns = [
            r'^[=-]+$',
            r'^>{3,}$',
            r'^<{3,}$',
            r'^={3,}$',
            r'^-{3,}$',
            r'^-------$',
            r'^=======$',
        ]
        
        end_patterns = [
            r'^[\]>]*\s*(?:REPLACE)\s*[\<\[ ]*$',
            r'^[>]+$',
            r'^[=]+$',
            r'^[-]+$',
        ]
        
        line_pattern = r':([a-z_]*line[a-z_]*):\s*(\d*)'
        
        # 1.1 Поиск стартового маркера (первая строка)
        start_marker = None
        if remaining_lines:
            first_line = remaining_lines[0].strip()
            for pattern in start_patterns:
                if re.match(pattern, first_line, re.IGNORECASE):
                    start_marker = remaining_lines.pop(0)
                    issues.append(f"Найден стартовый маркер: {start_marker[:50]}")
                    if debug:
                        print(f"{handlers.Colors.GREEN}✓ Стартовый маркер: {start_marker[:50]}{handlers.Colors.RESET}")
                    break
        
        # 1.2 Поиск маркера номера строки (строго следующая строка после стартового маркера)
        line_marker = None
        line_number = None
        
        if remaining_lines:
            next_line = remaining_lines[0].strip()
            match = re.search(line_pattern, next_line, re.IGNORECASE)
            if match:
                line_marker = remaining_lines.pop(0)
                line_number = match.group(2) if match.group(2) else None
                issues.append(f"Найден маркер строки: {line_marker[:50]}, значение: {line_number}")
                if debug:
                    print(f"{handlers.Colors.MAGENTA}✓ Маркер строки: {line_marker[:50]}{handlers.Colors.RESET}")
                
                # 1.3 Поиск разделителя после маркера строки (строго следующая строка)
                if remaining_lines:
                    next_next_line = remaining_lines[0].strip()
                    for pattern in separator_patterns:
                        if re.match(pattern, next_next_line):
                            separator_after_line = remaining_lines.pop(0)
                            issues.append(f"Найден разделитель после маркера: {separator_after_line}")
                            if debug:
                                print(f"{handlers.Colors.BLUE}✓ Разделитель после маркера: {separator_after_line}{handlers.Colors.RESET}")
                            break
        
        # 1.4 Поиск финального маркера (последняя строка)
        end_marker = None
        if remaining_lines:
            last_line = remaining_lines[-1].strip()
            for pattern in end_patterns:
                if re.match(pattern, last_line, re.IGNORECASE):
                    end_marker = remaining_lines.pop()
                    issues.append(f"Найден финальный маркер: {end_marker[:50]}")
                    if debug:
                        print(f"{handlers.Colors.RED}✓ Финальный маркер: {end_marker[:50]}{handlers.Colors.RESET}")
                    break
        
        # 1.5 Оставшиеся строки = чистый контент
        clean_content_lines = remaining_lines
        clean_content = '\n'.join(clean_content_lines)
        
        if debug:
            print(f"\n{handlers.Colors.YELLOW}Чистый контент (без маркеров):{handlers.Colors.RESET}")
            print(f"{handlers.Colors.YELLOW}{'-'*40}{handlers.Colors.RESET}")
            for i, line in enumerate(clean_content_lines[:10]):
                repr_line = line.replace(' ', '·').replace('\t', '→')
                print(f"{i:3}: '{repr_line[:80]}'")
            if len(clean_content_lines) > 10:
                print(f"... и еще {len(clean_content_lines) - 10} строк")
            print(f"{handlers.Colors.YELLOW}{'-'*40}{handlers.Colors.RESET}")
        
        # Фаза 2: Поиск в файле
        if not file_content:
            issues.append("Нет содержимого файла для верификации")
            return None, issues
        
        # Берем первые 3 строки из чистого контента (как есть, включая пустые, сохраняя пробелы)
        search_start_lines = clean_content_lines[:3]
        
        if not search_start_lines:
            issues.append("Чистый контент пуст")
            return None, issues
        
        if debug:
            print(f"\n{handlers.Colors.CYAN}Поиск в файле первых 3 строк SEARCH блока:{handlers.Colors.RESET}")
            for i, line in enumerate(search_start_lines):
                repr_line = line.replace(' ', '·').replace('\t', '→')
                print(f"  {i+1}: '{repr_line[:80]}'")
        
        # Ищем эти строки в файле
        file_lines_list = [file_content[k] for k in sorted(file_content.keys())]
        found_position = None
        whitespace_warning = False
        
        for i in range(len(file_lines_list) - len(search_start_lines) + 1):
            match = True
            trim_match = True
            
            for j, search_line in enumerate(search_start_lines):
                if i + j >= len(file_lines_list):
                    match = False
                    trim_match = False
                    break
                
                file_line = file_lines_list[i + j]
                
                # Основное сравнение: точное, с пробелами
                if file_line != search_line:
                    match = False
                    
                    # Дополнительное сравнение: trim версии (только для диагностики)
                    if file_line.strip() != search_line.strip():
                        trim_match = False
                
                # Если точное совпадение уже нарушено, можно не продолжать
                if not match and not trim_match:
                    break
            
            if match:
                found_position = i
                if debug:
                    print(f"{handlers.Colors.GREEN}✓ Найдено ТОЧНОЕ совпадение на позиции {i} (строка {i+1} в файле){handlers.Colors.RESET}")
                break
            
            # Если точного нет, но trim совпал - запоминаем для предупреждения
            if trim_match and not match:
                whitespace_warning = True
                if debug:
                    print(f"{handlers.Colors.YELLOW}⚠ На позиции {i}: trim версии совпадают, но есть различия в пробелах{handlers.Colors.RESET}")
                    for j, search_line in enumerate(search_start_lines[:2]):
                        if i + j < len(file_lines_list):
                            file_line = file_lines_list[i + j]
                            print(f"    Файл:   '{file_line.replace(' ', '·').replace('\t', '→')[:60]}'")
                            print(f"    Diff:   '{search_line.replace(' ', '·').replace('\t', '→')[:60]}'")
        
        if found_position is None:
            issues.append("Не удалось найти начало SEARCH блока в файле")
            if whitespace_warning:
                issues.append("ПРЕДУПРЕЖДЕНИЕ: Найдены совпадения trim-версий, но различия в пробелах")
            return None, issues
        
        # Фаза 3: Поиск истинного разделителя
        # Идем параллельно по файлу и чистому контенту до расхождения
        file_idx = found_position
        content_idx = 0
        true_separator = None
        true_separator_pos = None
        
        if debug:
            print(f"\n{handlers.Colors.CYAN}Поиск расхождения:{handlers.Colors.RESET}")
        
        while file_idx < len(file_lines_list) and content_idx < len(clean_content_lines):
            file_line = file_lines_list[file_idx]
            content_line = clean_content_lines[content_idx]
            
            if debug:
                file_repr = file_line.replace(' ', '·').replace('\t', '→')[:50]
                content_repr = content_line.replace(' ', '·').replace('\t', '→')[:50]
                print(f"  Сравнение: файл[{file_idx}]='{file_repr}' vs контент[{content_idx}]='{content_repr}'")
            
            if file_line != content_line:
                # Нашли расхождение
                if debug:
                    print(f"{handlers.Colors.YELLOW}⚠ Расхождение на позиции {content_idx}{handlers.Colors.RESET}")
                
                # Проверяем, есть ли разделитель в clean_content_lines на этой позиции
                if content_idx < len(clean_content_lines):
                    potential_separator = clean_content_lines[content_idx].strip()
                    for pattern in separator_patterns:
                        if re.match(pattern, potential_separator):
                            true_separator = potential_separator
                            true_separator_pos = content_idx
                            issues.append(f"Найден истинный разделитель: {true_separator} на позиции {true_separator_pos}")
                            if debug:
                                print(f"{handlers.Colors.GREEN}✓ Истинный разделитель: {true_separator}{handlers.Colors.RESET}")
                            break
                
                if true_separator is None:
                    issues.append(f"ОШИБКА: В месте расхождения нет разделителя (строка: '{clean_content_lines[content_idx][:50]}')")
                    return None, issues
                
                break
            
            file_idx += 1
            content_idx += 1
        
        if true_separator is None:
            issues.append("Не удалось найти истинный разделитель (достигнут конец файла или контента)")
            return None, issues
        
        # Фаза 4: Пересборка diff с экранированием ложных маркеров
        # Разделяем чистый контент на SEARCH и REPLACE блоки
        search_content_lines = clean_content_lines[:true_separator_pos]
        replace_content_lines = clean_content_lines[true_separator_pos + 1:] if true_separator_pos + 1 < len(clean_content_lines) else []
        
        if debug:
            print(f"\n{handlers.Colors.CYAN}Разделение контента:{handlers.Colors.RESET}")
            print(f"  SEARCH блок: {len(search_content_lines)} строк")
            print(f"  REPLACE блок: {len(replace_content_lines)} строк")
            print(f"  Истинный разделитель: '{true_separator}'")
        
        # Функция экранирования ложных маркеров
        def escape_fake_markers(lines_list: List[str]) -> List[str]:
            """Экранирует строки, похожие на маркеры, добавляя \ в начало"""
            fake_patterns = [
                r'^[\[<]*\s*(?:SEARCH|SOURCE|SRC)\s*[\>\] ]*$',
                r'^[\]>]*\s*(?:REPLACE)\s*[\<\[ ]*$',
                r'^[=-]{3,}$',
                r'^>{3,}$',
                r'^<{3,}$',
                r'^={3,}$',
                r'^-{3,}$',
                r'^-------$',
                r'^=======$',
                r':[a-z_]*line[a-z_]*:\s*\d*',
            ]
            
            escaped = []
            for line in lines_list:
                should_escape = False
                line_stripped = line.strip()
                for pattern in fake_patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        should_escape = True
                        break
                if should_escape and not line.startswith('\\'):
                    escaped.append('\\' + line)
                else:
                    escaped.append(line)
            return escaped
        
        search_content_escaped = escape_fake_markers(search_content_lines)
        replace_content_escaped = escape_fake_markers(replace_content_lines)
        
        # Собираем финальный diff
        final_lines = []
        
        # Стартовый маркер
        final_lines.append('<<<<<<< SEARCH')
        
        # Маркер строки (используем найденный или вычисляем)
        actual_line_num = found_position + 1  # 1-based
        if line_number:
            final_lines.append(f':start_line:{line_number}')
        else:
            final_lines.append(f':start_line:{actual_line_num}')
        
        # Разделитель после маркера
        final_lines.append('-------')
        
        # SEARCH контент
        final_lines.extend(search_content_escaped)
        
        # Разделитель между блоками
        final_lines.append('=======')
        
        # REPLACE контент
        final_lines.extend(replace_content_escaped)
        
        # Финальный маркер
        final_lines.append('>>>>>>> REPLACE')
        
        final_diff = '\n'.join(final_lines)
        
        if debug:
            print(f"\n{handlers.Colors.GREEN}{handlers.Colors.BOLD}Финальный diff после deep_search:{handlers.Colors.RESET}")
            self._print_diff_preview(final_lines, "Deep Search Result")
            print(f"\n{handlers.Colors.YELLOW}Проблемы и исправления:{handlers.Colors.RESET}")
            for issue in issues:
                print(f"  • {issue}")
        
        return final_diff, issues

    def _fix_apply_diff_tool(self, tc: dict, data: dict, debug: bool = False) -> List[str]:
        """
        Комплексное исправление apply_diff: форматирование и поиск start_line.
        
        Args:
            tc: Словарь с аргументами tool call (будет модифицирован)
            data: Данные файла {"type": "file_content", "content": {...}} (опционально)
            
        Returns:
            Tuple[bool, List[str]]: (были ли исправления, список сообщений об изменениях)
        """

        def print_structure(obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, dict):
                print(f"{prefix}dict with keys: {list(obj.keys())}")
                for key, value in obj.items():
                    print(f"{prefix}  '{key}': ", end="")
                    if isinstance(value, dict):
                        print(f"dict")
                        print_structure(value, indent + 2)
                    elif isinstance(value, list):
                        print(f"list[{len(value)}]")
                        print_structure(value, indent + 2)
                    else:
                        print(f"{type(value).__name__}")
            elif isinstance(obj, list):
                print(f"{prefix}list[{len(obj)}]")
                if obj:
                    print(f"{prefix}  first item type: {type(obj[0]).__name__}")
                    print_structure(obj[0], indent + 2)
            else:
                print(f"{prefix}{type(obj).__name__}")
        
        print("\n=== DEBUG: tc structure (no data) ===")
        print_structure(tc)
        print("======================================\n")

        changed = False
        final_diff = ""

        # ПРОВЕРКА ДО ЛЮБЫХ ИЗМЕНЕНИЙ: есть ли множественные разделители?
        diff = tc['function']['arguments']['diff']
        lines = diff.split('\n')
        separator_count = 0
        separator_pattern = re.compile(r'^[=<>-]{3,}$|^<<<<<<<|^>>>>>>>|^=======$|^-------$', re.IGNORECASE)
        
        for line in lines:
            if separator_pattern.match(line.strip()):
                separator_count += 1

        # Если разделителей слишком много (> 6) - вызываем deep_search
        if separator_count > 6:
            if debug:
                print(f"{handlers.Colors.YELLOW}⚠ Обнаружено {separator_count} разделителей, вызываем deep_search{handlers.Colors.RESET}")
            
            file_content = data.get("content", {}) if data and data.get("type") == "file_content" else {}
            
            final_diff, issues = self._fix_apply_diff_deep_search(diff, file_content, debug)
            
            if final_diff:
                tc['function']['arguments']['diff'] = final_diff
                changed = True
                return changed
            else:
                return changed

        # Проверяем, что ключи - int
        if not all(isinstance(k, int) for k in data['content'].keys()):
            logger.error(f"_fix_apply_diff_tool: ключи file_content должны быть int")
            logger.error(f"  tc args: {tc.get('function', {}).get('arguments', {})}")
            logger.error(f"  data keys type: {type(next(iter(data['content'].keys())))}")
            if debug:
                print(f"  ✗ Ошибка: ключи file_content должны быть int, получены {type(next(iter(data['content'].keys())))}")
            return changed

        diff = tc['function']['arguments']['diff']
        lines = diff.split('\n')
        step = 1
        # ИНИЦИАЛИЗИРУЕМ ГРАНИЦЫ SEARCH БЛОКА
        search_block = [0, 0]  # [start_line, length]

        print("===========INIT=============")
        print(f"Lines = {len(lines)}, Separators = {separator_count}")

        # 1. Исправляем формат diff (разделители, :start_line:, -------)
        # Шаг 1: Ищем любые разделители, меняем на "======="
        separator_patterns = [
            # Маркеры с SEARCH/SOURCE/SRC/REPLACE
            r'^[\[\]<>=-]*\s*(?:SEARCH|SOURCE|SRC|REPLACE)\s*[\[\]<>=-]*$',
            
            # Чистые разделители
            r'^[\[\]<>=-]+$',

            r'^:(start|end)_line:'
        ]
        
        
        separator_indices = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for pattern in separator_patterns:
                if re.match(pattern, line_stripped):
                    if debug and line_stripped != '=======':
                        print(f"  Заменяем разделитель на строке {i}: '{line_stripped}' -> '======='")
                    lines[i] = '======='
                    changed = True
                    separator_indices.append(i)
                    break
        
        if debug:
            print(f"--- ШАГ {step}: После замены разделителей на '=======' (найдено {len(separator_indices)} разделителей) ---")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)
        step += 1
        
        # Шаг 2: Меняем несколько подряд идущих разделителей на один
        i = 0
        while i < len(separator_indices) - 1:
            if separator_indices[i+1] == separator_indices[i] + 1:
                # Подряд идущие разделители - удаляем первый
                if debug:
                    print(f"  Удаляем дублирующийся разделитель на строке {separator_indices[i]}")
                del lines[separator_indices[i]]
                # Обновляем индексы: все последующие уменьшаем на 1
                for j in range(i+1, len(separator_indices)):
                    separator_indices[j] -= 1
                del separator_indices[i]
                changed = True
                # Не увеличиваем i, так как на этой позиции теперь новый разделитель
            else:
                i += 1
        
        if debug:
            print(f"--- ШАГ {step}: После удаления дублирующихся разделителей ---")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)
        step += 1
        
        # Шаг 3: Если в начале нет разделителя - добавляем
        if not separator_indices or separator_indices[0] != 0:
            if debug:
                print("  Добавляем разделитель в начало")
            lines.insert(0, '=======')
            separator_indices = [i+1 for i in separator_indices]
            separator_indices.insert(0, 0)
            changed = True
        
        if debug:
            print(f"--- ШАГ {step}: После добавления разделителя в начало ---")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)
        step += 1
        
        # Шаг 4: Если в конце нет разделителя - добавляем
        if not separator_indices or separator_indices[-1] != len(lines) - 1:
            if debug:
                print("  Добавляем разделитель в конец")
            lines.append('=======')
            separator_indices.append(len(lines) - 1)
            changed = True
        
        if debug:
            print(f"--- ШАГ {step}: После добавления разделителя в конец ---")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)
        step += 1
        
        # Шаг 5: Проверяем количество разделителей
        if debug:
            print(f"  Текущее количество разделителей: {len(separator_indices)}")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)
        
        if len(separator_indices) == 3:
            if debug:
                print("  Обнаружено 3 разделителя, добавляем :start_line: и разделитель")
            # После первого добавляем :start_line: и разделитель
            lines.insert(separator_indices[0] + 1, ':start_line:1')
            lines.insert(separator_indices[0] + 2, '-------')
            separator_indices = [
                separator_indices[0],
                separator_indices[0] + 2,
                separator_indices[1] + 2,
                separator_indices[2] + 2
            ]
            changed = True
        
        if debug:
            print(f"--- ШАГ {step}: После обработки случая с 3 разделителями ---")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)
        step += 1
        
        # Шаг 6: Проверяем количество разделителей
        if len(separator_indices) != 4:
            if debug:
                print(f"❌ Ошибка: найдено {len(separator_indices)} разделителей, нужно 4")
                print(f"   Индексы разделителей: {separator_indices}")
        
        if debug:
            print(f"  ✓ Найдено 4 разделителя на позициях: {separator_indices}")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)

        # Шаг 7: Заменяем разделители по порядку на правильные
        print("==============separator_indices===========")
        print(separator_indices)
        print("==========================================")
        # 1-й разделитель: <<<<<<< SEARCH
        old = lines[separator_indices[0]]
        lines[separator_indices[0]] = '<<<<<<< SEARCH'
        if debug:
            print(f"  Разделитель 1 (строка {separator_indices[0]}): '{old}' -> '<<<<<<< SEARCH'")
            self._print_diff_preview(lines, f"После шага {step}")
            print(separator_indices)
        
        # 2-й разделитель: -------
        old = lines[separator_indices[1]]
        lines[separator_indices[1]] = '-------'
        search_block[0] = separator_indices[1]+1
        if debug:
            print(f"  Разделитель 2 (строка {separator_indices[1]}): '{old}' -> '-------'")
            self._print_diff_preview(lines, f"После шага {step}")
        
        # 3-й разделитель: =======
        try:
            old = lines[separator_indices[2]]
            lines[separator_indices[2]] = '======='
            search_block[1] = separator_indices[2]-search_block[0]
        except Exception as e:
            pass
        if debug:
            try:
                print(f"  Разделитель 3 (строка {separator_indices[2]}): '{old}' -> '======='")
                self._print_diff_preview(lines, f"После шага {step}")
            except Exception:
                pass
        
        # 4-й разделитель: >>>>>>> REPLACE
        try:
            old = lines[separator_indices[3]]
            lines[separator_indices[3]] = '>>>>>>> REPLACE'
        except Exception as e:
            pass
        if debug:
            try:
                print(f"  Разделитель 4 (строка {separator_indices[3]}): '{old}' -> '>>>>>>> REPLACE'")
                self._print_diff_preview(lines, f"После шага {step}")
            except Exception:
                pass
        
        if debug:
            print(f"--- ШАГ {step}: После замены разделителей на правильные ---")
        # Удаляем пустые строки в конце REPLACE блока
        # (между 3-м и 4-м разделителями)
        try:
            j = separator_indices[3] - 1
            while j > separator_indices[2] and not lines[j].strip():
                del lines[j]
                separator_indices[3] -= 1
                j -= 1
                changed = True
        except Exception:
            pass

        
        # 2. ищем актуальный start_line
        if debug:
            print(f"--- Начинаем поиск актуального start_line ---")
            print(f"file_content type: {type(data.get('content')) if data else 'None'}")
        
        # Если есть данные файла, ищем актуальный start_line
        if data and data.get('content'):
            print(f"Есть файл")
            file_content = data['content']
            print(search_block)
            if search_block:
                print(f"Найдено содержимое search")
                # Ищем этот блок в файле
                actual_line = self._find_search_block_line(file_content, lines, search_block)
                if actual_line is not None:
                    print(f"Найдена стартовая строка = {actual_line}")
                    # Обновляем :start_line: в lines (списке строк)
                    for idx, line in enumerate(lines):
                        if line.strip().startswith(':start_line:'):
                            lines[idx] = f':start_line:{actual_line}'
                            if debug:
                                print(f"  ✓ Обновлен start_line на строке {idx}: {lines[idx]}")
                    
                    changed = True
                else:
                    if debug:
                        print(f"  ✗ SEARCH блок не найден в файле")
                    return False, ["retry_with_llm"]
            else:
                if debug:
                    print(f"  ✗ Не удалось извлечь SEARCH блок из diff")
        else:
            if debug:
                print(f"  ✗ Нет данных файла для поиска start_line")

        # Обновляем diff в tc
        tc['function']['arguments']['diff'] = '\n'.join(lines)

        # Собираем финальный diff для возврата
        final_diff = '\n'.join(lines)
        if debug:
            print(f"\n--- Финальный diff (первые 500 символов) ---")
            print(final_diff[:500])
            if len(final_diff) > 500:
                print(f"... и еще {len(final_diff) - 500} символов")
        
        return changed, final_diff

    def process_single_tool_call(self, tc: dict, index: int, data: dict = None) -> Tuple[bool, List[str]]:
        """Обрабатывает один tool call. Модифицирует tc напрямую если были изменения.
        
        Args:
            tc: Tool call для обработки
            index: Индекс tool call
            request_body: Тело запроса для извлечения file_lines (опционально)
        """
        
        handlers.validate_fields(
            func_name="AnswerProcessor.process_single_tool_call",
            obj_name=f"tool_call[{index}]",
            data=tc,
            fields=['function', 'id']  # id тоже полезно проверить
        )

        func = tc['function']
        handlers.validate_fields(
            func_name="AnswerProcessor.process_single_tool_call",
            obj_name=f"function in tool_call[{index}]",
            data=func,
            fields=['name', 'arguments']
        )

        tool_name = func['name']
        tool_call_id = tc['id']
        parsed_args = func['arguments']  # Уже распарсенный dict
        
        changes = []
        changed = False
        
        # 2. Исправляем tool calls
        try:
            if tool_name == "read_file":
                handlers.validate_fields(
                    func_name="AnswerProcessor.process_single_tool_call",
                    obj_name=f"arguments in tool_call[{index}]",
                    data=func['arguments'],
                    fields=['path']
                )
                if self._add_mode_slice_to_read_file(parsed_args):
                    changed = True
                    changes.append("read_file: добавлен mode=slice")
                if self._fix_read_file_offset(parsed_args):
                    changed = True
                    changes.append("read_file: offset исправлен с 0 на 1")
        
            if tool_name == "apply_diff":
                handlers.validate_fields(
                    func_name="AnswerProcessor.process_single_tool_call",
                    obj_name=f"arguments in tool_call[{index}]",
                    data=func['arguments'],
                    fields=['path', 'diff']
                )
                content_dict = data.get("content", {})
                if data and data.get("type") == "file_content":
                    logger.debug(f"   - content_dict type: {type(content_dict)}")
                    logger.debug(f"   - content_dict keys count: {len(content_dict.keys()) if content_dict else 0}")

                    if content_dict:
                        logger.debug(f"   ✓ Вызов _diff_fixer.fix...")
                        # ПЕРЕДАЕМ СЛОВАРЬ НАПРЯМУЮ
                        changed, final_diff = self._fix_apply_diff_tool(tc, data, True)
                        if changed:
                            logger.debug(f"   ✓ Исправление успешно!")
                        # ✅ ВАЛИДАЦИЯ ФОРМАТА ПОСЛЕ ИСПРАВЛЕНИЯ
                        diff = tc['function']['arguments'].get('diff', '')
                        if diff:
                            diff_lines = diff.split('\n')
                            is_valid, errors = self._diff_fixer.validate_diff_format(diff_lines)
                            
                            if not is_valid:
                                # Формируем сообщение для retry
                                error_msg = "Неверный формат diff. Обнаружены следующие проблемы:\n"
                                for error in errors[:5]:
                                    error_msg += f"• {error}\n"
                                error_msg += "\nПожалуйста, сформируйте корректный diff в формате:\n"
                                error_msg += "<<<<<<< SEARCH\n:start_line:1\n-------\n[старый код]\n=======\n[новый код]\n>>>>>>> REPLACE"
                                
                                retry_message = {
                                    "role": "user",
                                    "content": f"❌ Ошибка валидации apply_diff:\n\n{error_msg}"
                                }
                                return False, ["retry_with_llm"]
                        
                        return True, [f"apply_diff: исправлен"]
                    else:
                        logger.warning(f"   ❌ Нет данных function_content")
                        return False, [f"function_replace: нет данных файла"]
            
            if tool_name == "ask_followup_question":
                handlers.validate_fields(
                    func_name="AnswerProcessor.process_single_tool_call",
                    obj_name=f"arguments in tool_call[{index}]",
                    data=func['arguments'],
                    fields=['question', 'follow_up']
                )
                if self._fix_followup_question(parsed_args):
                    changed = True
                    changes.append("ask_followup_question: follow_up строка -> массив")
            
            if self._fix_empty_path(parsed_args, tool_name):
                changed = True
                changes.append(f"{tool_name}: пустой path заменен на '.'")
            
            # 3а. Конвертация function_replace → apply_diff
            if tool_name == "function_replace":
                handlers.validate_fields(
                    func_name="AnswerProcessor.process_single_tool_call",
                    obj_name=f"arguments in tool_call[{index}]",
                    data=func['arguments'],
                    fields=['path', 'function', 'full_code']
                )
                path = parsed_args.get("path", "")
                function_name = parsed_args.get("function", "")
                full_code = parsed_args.get("full_code", "")
                
                logger.debug(f"[DEBUG function_replace] Начало обработки")
                logger.debug(f"   - path: {path}")
                logger.debug(f"   - function_name: {function_name}")
                logger.debug(f"   - full_code length: {len(full_code)}")
                logger.debug(f"   - data is None: {data is None}")
                
                if data and data.get("type") == "function_content":
                    content_dict = data.get("content", {})
                    logger.debug(f"   - content_dict type: {type(content_dict)}")
                    logger.debug(f"   - content_dict keys count: {len(content_dict.keys()) if content_dict else 0}")
                    
                    if content_dict:
                        logger.debug(f"   ✓ Вызов _convert_function_replace...")
                        # ПЕРЕДАЕМ СЛОВАРЬ НАПРЯМУЮ
                        diff_result = self._convert_function_replace(path, function_name, full_code, content_dict)
                        
                        if diff_result:
                            logger.debug(f"   ✓ Конвертация успешна!")
                            tc['function']['name'] = 'apply_diff'
                            # Оставляем как dict, не сериализуем
                            tc['function']['arguments'] = {'path': path, 'diff': diff_result['diff']}
                            return True, [f"function_replace: конвертирован в apply_diff"]
                        else:
                            logger.warning(f"   ❌ _convert_function_replace вернул None")
                            return False, [f"function_replace: не удалось заменить '{function_name}'"]
                    else:
                        logger.warning(f"   ❌ content_dict пуст")
                        return False, [f"function_replace: нет данных функции '{function_name}' в {path}"]
                else:
                    logger.warning(f"   ❌ Нет данных function_content")
                    return False, [f"function_replace: нет данных функции"]
                                
        except Exception as e:
            changes.append(f"❌ FIX ERROR for {tool_name}: {e}")
            return False, changes
        
        # 4. Если были изменения - просто возвращаем
        if changed:
            return True, changes
        
        return False, changes

    def process(self, answer, data: str = None) -> Optional[dict]:
        """Обрабатывает ответ от LLM.
        
        Args:
            answer: Объект Answer с ответом от LLM
            data: Дополнительные данные, переданные из proxy.py
                  Формат: {"type": "file_content", "path": "...", "content": "..."}
            
        Returns:
            dict с action, если нужно выполнить дополнительное действие в proxy.py,
            иначе None
        """
        now = datetime.now().strftime("%H:%M:%S")
        self.reset()
        
        if (data == None):
            data = {"type":None, "path":None}
        
        # Проверяем обязательные поля в answer
        handlers.validate_fields(
            func_name="AnswerProcessor.process",
            obj_name="answer",
            data=answer,
            fields=['full_response', 'model', 'duration', 'status_code']
        )

        # Сохранение оригинального ответа теперь происходит в proxy.py (save_responce)
        
        # 1. Извлекаем tool calls из текста
        if self._extract_and_add_tool_calls(answer):
            self._was_changed = True
            self.changes_log.append("Извлечены tool calls из текста")
        
        # 3. Обрабатываем каждый tool call отдельно (модифицируем tc на месте)
        if answer.tool_calls:
            for i, tc in enumerate(answer.tool_calls):
                if tc['id'] not in self.progress:
                    self.progress[tc['id']] = "current"
                
                # Делаем глубокую копию tc
                tc_copy = copy.deepcopy(tc)
                tool_name = tc_copy['function']['name']

                # Парсим аргументы
                try:
                    parsed_args = self._parse_arguments(
                        tc_copy['function']['arguments'],
                        tc_copy['function']['name'],
                        tc_copy['id']
                    )
                    tc_copy['function']['arguments'] = parsed_args
                except handlers.ArgumentParseError as e:
                    # Логируем ошибку
                    logger.error(f"ОШИБКА ПАРСИНГА АРГУМЕНТОВ")
                    logger.error(f"   Tool: {e.tool_name}")
                    logger.error(f"   ID: {e.tool_call_id}")
                    logger.error(f"   Ошибка: {e}")
                    logger.error(f"   Аргументы: {e.original_args[:200] if e.original_args else 'None'}...")
                    
                    # Отправляем сообщение в LLM через существующий механизм retry
                    error_message = self._send_parse_error_to_llm(answer, e, tc['id'])
                    
                    # Возвращаем action для proxy.py
                    return {
                        'action': 'retry_with_llm',
                        'message': error_message,
                        'tool_call_id': tc['id']
                    }
                
                # Initialize variables before the conditional block
                changed = False
                changes = []
                
                if self.progress[tc['id']] == "current":
                    
                    if (tool_name == "function_replace"):
                        logger.debug(f"[DEBUG process] Ветка if: tool_name == function_replace")
                        func_path = tc_copy['function']['arguments']['path']
                        func_name = tc_copy['function']['arguments']['function']
                        logger.info(f"[request_function] Формируем запрос функции: path={func_path}, function_name={func_name}")
                        if data and data.get("type") == "function_content" and data.get("path") == func_path:
                            changed, changes = self.process_single_tool_call(tc_copy, i, data)
                            self.progress[tc['id']] = "completed"
                        else:
                            logger.info(f"[request_function] Возврат action: path={func_path}, function_name={func_name}, data={data}")
                            return {
                                'action': 'request_function',
                                'path': func_path,
                                'function_name': func_name
                            }
                    elif (tool_name == "apply_diff"):
                        logger.debug(f"[DEBUG process] Ветка elif: tool_name == apply_diff")
                        diff_path = tc_copy['function']['arguments']['path']
                        if data and data.get("type") == "file_content" and data.get("path") == diff_path:
                            changed, changes = self.process_single_tool_call(tc_copy, i, data)

                            # ✅ Проверяем сигнал retry
                            if "retry_with_llm" in changes:
                                return {
                                    'action': 'retry_with_llm',
                                    'message': {
                                        "role": "user",
                                        "content": "❌ Неверный формат apply_diff. Пожалуйста, исправьте."
                                    },
                                    'tool_call_id': tc['id']
                                }
                            
                            self.progress[tc['id']] = "completed"
                        else:
                            logger.info(f"[request_file] Возврат action: path={diff_path}, data={data}")
                            return {
                                'action': 'request_file',
                                'path': diff_path
                            }
                    else:
                        changed, changes = self.process_single_tool_call(tc_copy, i)
                        self.progress[tc['id']] = "completed"
                
                    # Сериализуем аргументы обратно в строку и обновляем tool_calls
                    tc_copy['function']['arguments'] = self._serialize_arguments(tc_copy['function']['arguments'])
                    answer.tool_calls[i] = copy.deepcopy(tc_copy)
                    
                    self.progress[tc['id']] = "completed"

                if changed:
                    self._was_changed = True
                    self.changes_log.extend(changes)
            
            # Обновляем full_response если были изменения
            if self._was_changed:
                if "choices" not in answer.full_response:
                    raise ValueError("full_response missing 'choices' field")
                
                if not answer.full_response["choices"]:
                    raise ValueError("'choices' array is empty")
                
                if "message" not in answer.full_response["choices"][0]:
                    answer.full_response["choices"][0]["message"] = {}
                
                answer.full_response["choices"][0]["message"]["tool_calls"] = answer.tool_calls
        
        # 4. Декодируем Unicode escape последовательности
        if self._was_changed:
            answer.full_response = self._decode_all_strings(answer.full_response)
        
        # 5. Валидация всех tool calls
        is_valid, validation_errors = self.validate_tool_calls(answer)
        if not is_valid:
            logger.error(f"Tool call validation failed: {validation_errors}")
            for error in validation_errors:
                error_tool_name = error.get('tool_name')
                error_message = error.get('message')
                logger.error(f"   - {error_tool_name}: {error_message}")
        
        # 7. Выводим статистику
        if self._was_changed:
            for change in self.changes_log:
                logger.info(f"      - {change}")
        
        return None

    def _prepare_retry_request(self, answer) -> None:
        """
        Подготавливает повторный запрос к LLM.
        """
        if not hasattr(answer, 'retry_requests') or not answer.retry_requests:
            return
        
        # Формируем системное сообщение для retry
        retry_messages = []
        
        for retry in answer.retry_requests:
            msg = retry['message']
            tool_call_id = retry['tool_call_id']
            attempt = retry['attempt']
            
            # Формируем сообщение для LLM
            retry_prompt = f"""
[ПОВТОРНЫЙ ЗАПРОС #{attempt}]

При обработке предыдущего запроса возникли проблемы:

{msg['title']}
{msg['message']}

Детали:
{chr(10).join(msg['details'])}

Рекомендация:
{msg['advice']}

Пожалуйста, исправьте указанные проблемы и пришлите исправленный apply_diff.
"""
            
            retry_messages.append({
                "role": "system",
                "content": retry_prompt,
                "tool_call_id": tool_call_id
            })
        
        # Сохраняем в answer для отправки
        answer.retry_prompt = retry_messages
    
    def _args_to_dict(self, args_str: str, tool_name: str = "unknown", tool_call_id: str = "unknown") -> Dict[str, Any]:
        """
        Конвертирует Python-подобный текст аргументов в dict.
        
        Поддерживаемые форматы:
        - JSON строка: '{"path": "file.py", "mode": "slice"}'
        - Python-подобный: 'path="file.py", mode="slice"'
        - Mixed: 'path="file.py", recursive=True'
        
        Returns:
            Dict с распарсенными аргументами
        """
        if isinstance(args_str, dict):
            return args_str
        
        if not isinstance(args_str, str):
            raise handlers.ArgumentParseError(
                f"Неверный тип аргументов: ожидается str или dict, получен {type(args_str)}",
                tool_name=tool_name, tool_call_id=tool_call_id, original_args=str(args_str)
            )
        
        args_str = args_str.strip()
        if not args_str:
            return {}
        
        # 1. Пробуем как JSON
        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # 2. Пробуем распарсить Python-подобный синтаксис: key=value, key2="value2"
        result = {}
        # Используем более надежную регулярку для парсинга параметров
        # Находим все key=value пары, учитывая вложенные структуры в строках
        remaining = args_str
        while remaining.strip():
            # Ищем имя параметра (слово с возможными подчеркиваниями и цифрами)
            key_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', remaining.strip())
            if not key_match:
                raise handlers.ArgumentParseError(
                    f"Невозможно распарсить аргументы: ожидается 'key=value', получено '{remaining[:50]}'",
                    tool_name=tool_name, tool_call_id=tool_call_id, original_args=args_str
                )
            
            key = key_match.group(1)
            # Сдвигаем позицию после key=
            pos = key_match.end()
            rest = remaining.strip()[pos:]
            
            # Теперь нужно извлечь значение (до следующего "," или конца строки)
            # Значение может быть: строкой, boolean, числом, или списком/словарем
            value, consumed = self._parse_single_value(rest)
            result[key] = value
            
            # Сдвигаем позицию
            remaining = remaining.strip()[pos + consumed:].strip()
            
            # Пропускаем запятую, если есть
            if remaining.startswith(','):
                remaining = remaining[1:].strip()
        
        return result
    
    def _parse_single_value(self, text: str) -> tuple[Any, int]:
        """
        Парсит одиночное значение из начала строки.
        
        Returns:
            (значение, количество_символов_прочитано)
        """
        text = text.lstrip()
        if not text:
            raise ValueError("Пустой текст для парсинга значения")
        
        # Строка в двойных кавычках
        if text.startswith('"'):
            # Ищем закрывающую кавычку, учитывая экранирование
            i = 1
            while i < len(text):
                if text[i] == '\\':
                    i += 2  # Пропускаем экранированный символ
                    continue
                if text[i] == '"':
                    value = text[:i+1]
                    # Декодируем строку (убираем кавычки и обрабатываем экранирование)
                    try:
                        parsed = json.loads(value)
                        return parsed, i + 1
                    except json.JSONDecodeError:
                        # Если не получилось, возвращаем как есть (без кавычек)
                        return value[1:i], i + 1
                i += 1
            raise ValueError(f"Незакрытая строка: {text[:50]}")
        
        # Строка в одинарных кавычках
        if text.startswith("'"):
            i = 1
            while i < len(text):
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == "'":
                    value = text[1:i]
                    return value, i + 1
                i += 1
            raise ValueError(f"Незакрытая строка: {text[:50]}")
        
        # Boolean или null
        for lit in ('True', 'False', 'None'):
            if text.startswith(lit):
                if len(text) == len(lit) or not text[len(lit)].isalnum() and text[len(lit)] != '_':
                    if lit == 'True':
                        return True, len(lit)
                    elif lit == 'False':
                        return False, len(lit)
                    elif lit == 'None':
                        return None, len(lit)
        
        # Число (int или float)
        num_match = re.match(r'(-?\d+\.?\d*)(?:[eE][+-]?\d+)?', text)
        if num_match:
            num_str = num_match.group(1)
            try:
                if '.' in num_str or 'e' in num_str.lower():
                    return float(num_str), len(num_str)
                else:
                    return int(num_str), len(num_str)
            except ValueError:
                pass
        
        # Список или словарь (JSON-подобные)
        if text.startswith('[') or text.startswith('{'):
            try:
                parsed = json.loads(text)
                # Найти, где заканчивается этот JSON
                import json as json_module
                decoder = json_module.JSONDecoder()
                obj, end = decoder.raw_decode(text)
                return obj, end
            except json.JSONDecodeError:
                raise ValueError(f"Невалидный JSON: {text[:50]}")
        
        # Если ничего не подошло - это просто слово/идентификатор
        word_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', text)
        if word_match:
            return word_match.group(1), len(word_match.group(1))
        
        raise ValueError(f"Неизвестный формат значения: {text[:50]}")

    def _extract_and_add_tool_calls(self, answer) -> bool:
        if not answer.content or not isinstance(answer.content, str):
            return False
        
        found = False
        
        # Markdown блоки
        pattern_block = r'```tool_code\s*\n?\s*\[\s*([a-zA-Z0-9_]+)\s*\((.*?)\)\s*\]\s*```'
        
        for match in re.finditer(pattern_block, answer.content, flags=re.DOTALL):
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Парсим в dict (для внутреннего использования)
            args_dict = self._args_to_dict(params_str)
            # Сериализуем в JSON строку, чтобы сохранять как string
            args_str = self._serialize_arguments(args_dict)
           
            answer.tool_calls.append({
               "id": f"call_text_{int(time.time())}_{len(answer.tool_calls)}",
               "type": "function",
               "function": {
                   "name": tool_name.strip(),
                   "arguments": args_str  # сохраняем как JSON строку
               }
            })
            
            found = True
        
        answer.content = re.sub(pattern_block, '', answer.content, flags=re.DOTALL)
        
        # Inline вызовы
        pattern_inline = r'\[\s*([a-zA-Z0-9_]+)\s*\((.*?)\)\s*\]'
        
        for match in re.finditer(pattern_inline, answer.content):
            tool_name = match.group(1)
            params_str = match.group(2)
            
            args_dict = self._args_to_dict(params_str)
            # Сериализуем в JSON строку
            args_str = self._serialize_arguments(args_dict)
            
            answer.tool_calls.append({
                "id": f"call_text_{int(time.time())}_{len(answer.tool_calls)}",
                "type": "function",
                "function": {
                    "name": tool_name.strip(),
                    "arguments": args_str
                }
            })
            
            found = True
        
        answer.content = re.sub(pattern_inline, '', answer.content)
        answer.content = re.sub(r'\n\s*\n\s*\n', '\n\n', answer.content)
        answer.content = answer.content.strip()
        
        return found

    def _fix_followup_question(self, parsed_args: dict) -> bool:
        """
        Исправляет follow_up в ask_followup_question.
        
        Преобразует:
        1. Строку с JSON-массивом -> массив объектов
        2. Массив строк -> массив объектов {text: "...", mode: null}
        """
        if "follow_up" not in parsed_args:
            return False
        
        follow_up = parsed_args["follow_up"]
        
        # Case 1: это строка с JSON
        if isinstance(follow_up, str):
            try:
                parsed = json.loads(follow_up)
                if isinstance(parsed, list):
                    parsed_args["follow_up"] = self._normalize_follow_up_array(parsed)
                    return True
            except json.JSONDecodeError:
                pass
        
        # Case 2: это массив строк
        elif isinstance(follow_up, list) and follow_up and isinstance(follow_up[0], str):
            parsed_args["follow_up"] = self._normalize_follow_up_array(follow_up)
            return True
        
        return False

    def _normalize_follow_up_array(self, arr: list) -> list:
        """Преобразует массив строк в массив объектов {text: "...", mode: null}"""
        return [{"text": item, "mode": None} for item in arr]

    def _add_mode_slice_to_read_file(self, parsed_args: dict) -> bool:
        """
        Добавляет mode=slice в read_file.
        
        Args:
            parsed_args: Dict с аргументами (уже распарсенными)
            
        Returns:
            True если были изменения, иначе False
        """
        if "mode" not in parsed_args:
            parsed_args["mode"] = "slice"
            return True
        return False

    def _fix_read_file_offset(self, parsed_args: dict) -> bool:
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

    def _fix_empty_path(self, parsed_args: dict, tool_name: str) -> bool:
        """
        Исправляет пустой path (заменяет на '.').
        
        Args:
            parsed_args: Dict с аргументами (уже распарсенными)
            tool_name: Имя инструмента
            
        Returns:
            True если были изменения, иначе False
        """
        tools_with_path = ["search_files", "list_files", "read_file", "apply_diff", "write_to_file", "function_replace"]
        
        if tool_name not in tools_with_path:
            return False
        
        if "path" in parsed_args and (parsed_args["path"] == "" or parsed_args["path"] is None):
            parsed_args["path"] = "."
            return True
        
        return False

    def _is_valid_format(self, diff: str) -> bool:
        """Проверяет, является ли формат diff правильным"""
        if "<<<<<<< SEARCH" not in diff or ">>>>>>> REPLACE" not in diff:
            return False
        
        lines = diff.split('\n')
        
        # Проверяем наличие :start_line:
        has_start_line = False
        for line in lines:
            if re.search(r':start_line:\s*\d+', line):
                has_start_line = True
                break
        
        if not has_start_line:
            return False
        
        # Проверяем наличие ------- после :start_line:
        found_dash = False
        for line in lines:
            if line.strip() == "-------":
                found_dash = True
                break
        
        # Проверяем наличие ======= между блоками
        found_sep = False
        for line in lines:
            if line.strip() == "=======":
                found_sep = True
                break
        
        return found_dash and found_sep

    def _normalize_indent(self, text: str, remove_all: bool = False, preserve_first_line: bool = False) -> str:
        """
        Нормализует отступы в тексте.
        
        Args:
            text: Текст для нормализации
            remove_all: Если True, убирает все отступы полностью
            preserve_first_line: Если True, сохраняет отступ первой строки
        """
        lines = text.split('\n')
        if not lines:
            return text
        
        if remove_all:
            # Убираем все отступы
            result = []
            for line in lines:
                result.append(line.lstrip())
            return '\n'.join(result)
        
        # Находим минимальный отступ среди непустых строк
        min_indent = float('inf')
        for line in lines:
            stripped = line.strip()
            if stripped:  # только непустые строки
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        
        # Если все строки пустые или нет отступов, возвращаем как есть
        if min_indent == float('inf') or min_indent == 0:
            return text
        
        # Убираем минимальный отступ из всех строк
        result = []
        for i, line in enumerate(lines):
            if line.strip():
                if preserve_first_line and i == 0:
                    result.append(line)  # сохраняем отступ первой строки
                else:
                    result.append(line[min_indent:])
            else:
                result.append('')
        
        return '\n'.join(result)

    def _find_search_block_line(self, file_content: Dict[int, str], lines: List[str], search_block: tuple) -> Optional[int]:
        """
        Находит номер строки, где начинается SEARCH блок в файле.
        
        Args:
            file_content: Словарь {номер_строки: содержимое}
            lines: Список строк diff (уже разбитый на строки)
            search_block: Кортеж (start_line, length) - начало и длина SEARCH блока в diff
            
        Returns:
            Номер строки в файле (1-based) или None если не найден
        """
        # Проверяем, что ключи - int
        if not all(isinstance(k, int) for k in file_content.keys()):
            logger.error(f"_find_search_block_line: ключи file_content должны быть int, получены {type(next(iter(file_content.keys())))}")
            return None

        if not file_content or not lines or not search_block:
            logger.debug("_find_search_block_line: нет file_content, lines или search_block")
            return None
            
        # Распаковываем search_block
        if isinstance(search_block, (list, tuple)) and len(search_block) == 2:
            start_line, block_length = search_block
        else:
            logger.error(f"_find_search_block_line: search_block должен быть кортежем (start, length), получен {type(search_block)}")
            return None
        
        # Проверяем границы
        if start_line < 0 or start_line >= len(lines):
            logger.debug(f"_find_search_block_line: start_line {start_line} вне диапазона diff (0-{len(lines)-1})")
            return None
        
        # Получаем строки файла в правильном порядке
        sorted_keys = sorted(file_content.keys())
        file_lines = [file_content[k] for k in sorted_keys]
        line_nums = sorted_keys
        
        logger.debug(f"_find_search_block_line: поиск блока из {block_length} строк в файле из {len(file_lines)} строк")
        logger.debug(f"_find_search_block_line: блок в diff находится на строках {start_line}-{start_line + block_length - 1}")

        # Словарь подмен символов
        char_replacements = {
            '"': '«',
            "'": '′',
            '-': '—',
            '«': '"',
            '′': "'",
            '—': '-'
        }
        
        # Функция для нормализации строки (strip и удаление префикса "| ")
        def normalize_line(line: str) -> str:
            if line.startswith('| '):
                line = line[2:]
            return line.strip()
        
        print(f"_find_search_block_line: поиск блока из {block_length} строк в файле из {len(file_lines)} строк")
        print(f"_find_search_block_line: блок в diff находится на строках {start_line}-{start_line + block_length - 1}")

        # Ищем совпадение в файле
        for i in range(len(file_lines) - block_length + 1):
            print(f"=== Поиск в файле: позиция {i} ===")
            match = True
            matched_indices = []  # Сохраняем индексы строк, которые совпали
            
            for j in range(block_length):
                print(f"--- Строка {j} блока ---")
                if i + j >= len(file_lines):
                    print(f"Выход за границы")
                    match = False
                    break

                file_line = file_lines[i + j]
                search_line = lines[start_line + j]
                
                # Нормализуем строки для сравнения
                file_line_normalized = file_line.strip()
                search_line_normalized = normalize_line(search_line)
                
                # Пустые строки считаются совпадающими
                if file_line_normalized == '' and search_line_normalized == '':
                    print(f"Пустые строки")
                    matched_indices.append(j)
                    continue
                
                # Сначала прямое сравнение нормализованных строк
                if file_line_normalized == search_line_normalized:
                    print(f"Прямое совпадение после нормализации")
                    print(f"File = '{file_line_normalized}'")
                    print(f"Diff = '{search_line_normalized}'")
                    matched_indices.append(j)
                    continue
                
                print(f"Несовпадение, пробуем подмену символов")
                print(f"  файл: {file_line_normalized[:50]}")
                print(f"  diff: {search_line_normalized[:50]}")
                
                # Пытаемся добиться совпадения через подмену символов
                attempt = 0
                max_attempts = 10
                current_line = search_line_normalized
                
                while attempt < max_attempts and file_line_normalized != current_line:
                    print(f"Попытка {attempt + 1}")
                    
                    # Находим позицию первого несовпадающего символа
                    min_len = min(len(file_line_normalized), len(current_line))
                    mismatch_pos = None
                    for pos in range(min_len):
                        if file_line_normalized[pos] != current_line[pos]:
                            mismatch_pos = pos
                            break
                    
                    if mismatch_pos is None:
                        if len(file_line_normalized) != len(current_line):
                            print(f"Разная длина строк: файл={len(file_line_normalized)}, diff={len(current_line)}")
                        break
                    
                    diff_char = current_line[mismatch_pos]
                    file_char = file_line_normalized[mismatch_pos]
                    print(f"Несовпадение на позиции {mismatch_pos}: diff='{diff_char}' vs file='{file_char}'")
                    
                    # Проверяем подмену
                    replaced = False
                    for s1, s2 in char_replacements.items():
                        if diff_char == s1 and file_char == s2:
                            print(f"Подмена: '{s1}' -> '{s2}'")
                            # Заменяем символ в оригинальной строке lines (не в нормализованной)
                            line_list = list(lines[start_line + j])
                            # Находим позицию в оригинальной строке (может отличаться из-за "| " и пробелов)
                            orig_pos = mismatch_pos
                            if lines[start_line + j].startswith('| '):
                                orig_pos = mismatch_pos + 2
                            if orig_pos < len(line_list):
                                line_list[orig_pos] = s2
                                lines[start_line + j] = ''.join(line_list)
                                current_line = normalize_line(lines[start_line + j])
                                replaced = True
                                break
                    
                    if not replaced:
                        print(f"Нет подходящей подмены")
                        break
                    
                    attempt += 1
                
                if file_line_normalized == current_line:
                    print(f"Подмена успешна, строка обновлена")
                    self._was_changed = True
                    matched_indices.append(j)
                    continue
                
                print(f"Не удалось добиться совпадения")
                match = False
                break
            
            # Если все строки блока совпали
            if match and len(matched_indices) == block_length:
                result_line = line_nums[i]
                print(f"Блок найден! Стартовая строка: {result_line}")
                
                # Заменяем все строки search_block в diff на реальные строки из файла
                print(f"Заменяем строки {start_line}-{start_line + block_length - 1} в diff на строки из файла")
                for j in range(block_length):
                    original_file_line = file_lines[i + j]
                    old_diff_line = lines[start_line + j]
                    lines[start_line + j] = original_file_line
                    print(f"  Строка {j}: '{old_diff_line[:50]}' -> '{original_file_line[:50]}'")
                    self._was_changed = True
                
                return result_line
        
        print(f"Блок не найден")
        return None

    def _send_parse_error_to_llm(self, answer, error: handlers.ArgumentParseError, tool_call_id: str) -> Dict:
        """
        Формирует сообщение для отправки в LLM при ошибке парсинга аргументов.
        
        Args:
            answer: Объект Answer
            error: Исключение handlers.ArgumentParseError
            tool_call_id: ID tool call, который не удалось распарсить
            
        Returns:
            Dict с сообщением для добавления в историю
        """
        # Простое дружественное сообщение
        error_message = f"""❌ Не удалось обработать ваш tool call `{error.tool_name}`. Пожалуйста, попробуйте сформулировать запрос по-другому.

    **Проблема:** {str(error)}

    **Что делать:** Проверьте синтаксис аргументов и попробуйте еще раз.
    """

        # Логируем ошибку
        timestamp = datetime.now().isoformat()
        logger.error(f"ОШИБКА ПАРСИНГА АРГУМЕНТОВ")
        logger.error(f"📅 Время: {timestamp}")
        logger.error(f"🔧 Tool: {error.tool_name}")
        logger.error(f"🆔 Tool Call ID: {error.tool_call_id}")
        logger.error(f"📝 Ошибка: {error}")
        logger.error(f"📄 Аргументы (первые 500 символов): {error.original_args[:500] if error.original_args else 'None'}...")
        logger.error(f"🤖 Модель: {answer.model}")
        logger.error(f"📤 Отправляем запрос на переформулирование в LLM...")

        # Создаем сообщение для добавления в историю
        retry_message = {
            "role": "user",
            "content": error_message
        }

        return retry_message

    def _print_diff_preview(self, lines: List[str], title: str) -> None:
        """Выводит preview diff после каждого шага с цветными маркерами"""
        print(f"\n{handlers.Colors.CYAN}{handlers.Colors.BOLD}{title}{handlers.Colors.RESET}")
        print(f"{handlers.Colors.YELLOW}{'='*60}{handlers.Colors.RESET}")
        for i, line in enumerate(lines):
            if line.startswith('<<<<<<< SEARCH'):
                print(f"{handlers.Colors.GREEN}{line}{handlers.Colors.RESET}")
            elif line.startswith('======='):
                print(f"{handlers.Colors.YELLOW}{line}{handlers.Colors.RESET}")
            elif line.startswith('>>>>>>> REPLACE'):
                print(f"{handlers.Colors.RED}{line}{handlers.Colors.RESET}")
            elif line.startswith('-------'):
                print(f"{handlers.Colors.BLUE}{line}{handlers.Colors.RESET}")
            elif line.startswith(':start_line:'):
                print(f"{handlers.Colors.MAGENTA}{line}{handlers.Colors.RESET}")
            else:
                print(f"{handlers.Colors.RESET}{line}{handlers.Colors.RESET}")

    def _print_diff_pretty(self, diff: str) -> None:
        """Выводит финальный diff в красивом формате"""
        print(f"\n{handlers.Colors.BOLD}{handlers.Colors.CYAN}{'='*60}")
        print(f"ФИНАЛЬНЫЙ DIFF")
        print(f"{'='*60}{handlers.Colors.RESET}")
        lines = diff.split('\n')
        for line in lines:
            if line.startswith('<<<<<<< SEARCH'):
                print(f"{handlers.Colors.GREEN}{line}{handlers.Colors.RESET}")
            elif line.startswith('======='):
                print(f"{handlers.Colors.YELLOW}{line}{handlers.Colors.RESET}")
            elif line.startswith('>>>>>>> REPLACE'):
                print(f"{handlers.Colors.RED}{line}{handlers.Colors.RESET}")
            elif line.startswith('-------'):
                print(f"{handlers.Colors.BLUE}{line}{handlers.Colors.RESET}")
            elif line.startswith(':start_line:'):
                print(f"{handlers.Colors.MAGENTA}{line}{handlers.Colors.RESET}")
            else:
                print(f"{handlers.Colors.RESET}{line}{handlers.Colors.RESET}")
        print(f"{handlers.Colors.CYAN}{'='*60}{handlers.Colors.RESET}")

    def validate_diff_format(self, diff_lines: List[str]) -> Tuple[bool, List[str]]:
        """
        Валидирует формат diff.
        
        Ожидаемый формат:
        1. <<<<<<< SEARCH
        2. :start_line:N (где N - валидное число, и после числа ничего нет)
        3. -------
        4. ... search content ...
        5. =======
        6. ... replace content ...
        7. >>>>>>> REPLACE
        
        ======= не может быть на 4 строке (индекс 3)
        """
        errors = []
        
        if not diff_lines:
            errors.append("Diff пуст")
            return False, errors
        
        if diff_lines[0] != '<<<<<<< SEARCH':
            errors.append(f"Строка 1: ожидается '<<<<<<< SEARCH', получено '{diff_lines[0]}'")
        
        if not diff_lines[1].startswith(':start_line:'):
            errors.append(f"Строка 2: ожидается ':start_line:N', получено '{diff_lines[1]}'")
        else:
            number_part = diff_lines[1][len(':start_line:'):]
            if not number_part.isdigit():
                errors.append(f"Строка 2: после ':start_line:' ожидается число, получено '{number_part}'")
        
        if diff_lines[2] != '-------':
            errors.append(f"Строка 3: ожидается '-------', получено '{diff_lines[2]}'")
        
        separator_idx = -1
        for i, line in enumerate(diff_lines):
            if line == '=======':
                separator_idx = i
                break
        
        if separator_idx == -1:
            errors.append("Отсутствует разделитель '======='")
        else:
            if separator_idx == 3:
                errors.append("Разделитель '=======' не может быть на 4 строке (должен быть хотя бы одна строка SEARCH контента)")

        if diff_lines[-1] != '>>>>>>> REPLACE':
            errors.append(f"Последняя строка: ожидается '>>>>>>> REPLACE', получено '{diff_lines[-1]}'")
        
        return len(errors) == 0, errors
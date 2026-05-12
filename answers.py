# answers.py
import json
import re
import time
import copy
import os
import yaml
import handlers
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from log import log_response, log_retry_attempt


class ArgumentParseError(Exception):
    """Исключение при ошибке парсинга аргументов tool call"""
    def __init__(self, message: str, tool_name: str = None, tool_call_id: str = None, original_args: str = None):
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.original_args = original_args
        super().__init__(message)
        

class RetryManager:
    """
    Управляет повторными попытками отправки запросов к LLM.
    """
    
    def __init__(self, max_retries: int = 2):
        """
        Инициализация менеджера retry.
        
        Args:
            max_retries: Максимальное количество повторных попыток (по умолчанию 2)
        """
        self.max_retries = max_retries
        self.retry_history = {}  # history by conversation_id or tool_call_id
    
    def should_retry(self, conversation_id: str, tool_call_id: str) -> bool:
        """
        Проверяет, нужно ли делать еще одну попытку.
        
        Args:
            conversation_id: ID диалога
            tool_call_id: ID tool call
            
        Returns:
            True если нужно retry, False если лимит исчерпан
        """
        key = f"{conversation_id}:{tool_call_id}"
        history = self.retry_history.get(key, [])
        return len(history) < self.max_retries
    
    def register_attempt(self, conversation_id: str, tool_call_id: str, 
                        errors: List[Dict], retry_message: Dict) -> int:
        """
        Регистрирует попытку retry.
        
        Args:
            conversation_id: ID диалога
            tool_call_id: ID tool call
            errors: Список ошибок, вызвавших retry
            retry_message: Сообщение, отправленное в retry
            
        Returns:
            Номер попытки (1-based)
        """
        key = f"{conversation_id}:{tool_call_id}"
        if key not in self.retry_history:
            self.retry_history[key] = []
        
        attempt_number = len(self.retry_history[key]) + 1
        
        self.retry_history[key].append({
            "attempt": attempt_number,
            "timestamp": datetime.now().isoformat(),
            "errors": errors,
            "retry_message": retry_message
        })
        
        # Логируем попытку
        log_retry_attempt(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            attempt=attempt_number,
            errors=errors,
            retry_message=retry_message
        )
        
        return attempt_number
    
    def get_attempts(self, conversation_id: str, tool_call_id: str) -> List[Dict]:
        """Возвращает историю попыток для конкретного tool call"""
        key = f"{conversation_id}:{tool_call_id}"
        return self.retry_history.get(key, [])
    
    def get_attempt_count(self, conversation_id: str, tool_call_id: str) -> int:
        """Возвращает количество сделанных попыток"""
        return len(self.get_attempts(conversation_id, tool_call_id))
    
    def should_abort(self, conversation_id: str, tool_call_id: str) -> bool:
        """
        Проверяет, нужно ли прервать выполнение (лимит исчерпан).
        
        Returns:
            True если нужно прервать и отдать как есть
        """
        return self.get_attempt_count(conversation_id, tool_call_id) >= self.max_retries


class AnswerProcessor:
    """
    Класс для обработки и исправления tool calls от LLM.
    process() ИЗМЕНЯЕТ исходные данные напрямую.
    Свойство changed показывает, были ли изменения.
    
    Логика работы с apply_diff:
    1. СНАЧАЛА исправляем формат (fix_apply_diff) - должен привести к правильному формату
    2. ПОТОМ валидируем (validate_apply_diff) - обнаруживаем ВСЕ проблемы
    3. Классифицируем проблемы:
       - Проблемы с содержимым → RETRY (с notice)
       - Проблемы с форматированием → баг в fix_apply_diff! Логируем и RETRY
    4. Максимум 2 попытки retry, после чего отдаем как есть
    """
    
    # Шаблоны сообщений для retry
    RETRY_MESSAGES = {
        # Проблемы с содержимым - понятные пользователю
        "content": {
            "content_mismatch": {
                "title": "❌ Текст для замены не найден",
                "message": "Указанный фрагмент не найден в файле. Возможно, файл изменился или был указан неправильный контекст.",
                "advice": "Пожалуйста, прочитайте актуальное содержимое файла и укажите правильный SEARCH блок.",
                "priority": 1
            },
            "file_not_found": {
                "title": "📁 Файл не найден",
                "message": "Файл по указанному пути не существует.",
                "advice": "Проверьте путь к файлу или создайте его отдельной операцией.",
                "priority": 1
            },
            "range_out_of_bounds": {
                "title": "📏 Некорректный диапазон строк",
                "message": "Указанный диапазон строк выходит за границы файла.",
                "advice": "Проверьте актуальный размер файла и скорректируйте номера строк.",
                "priority": 2
            },
            "identical_blocks": {
                "title": "🔄 Блоки идентичны",
                "message": "SEARCH и REPLACE блоки完全相同 - изменения не будут применены.",
                "advice": "Укажите, что именно нужно изменить в REPLACE блоке.",
                "priority": 2
            },
            "file_read_error": {
                "title": "🔒 Ошибка чтения файла",
                "message": "Не удалось прочитать файл (проблемы с кодировкой или правами доступа).",
                "advice": "Проверьте права доступа к файлу и его кодировку.",
                "priority": 3
            },
            "no_workspace": {
                "title": "⚙️ Техническая ошибка",
                "message": "Не удалось определить рабочую директорию.",
                "advice": "Пожалуйста, повторите запрос позже.",
                "priority": 3,
                "internal": True
            }
        },
        
        # Проблемы с форматированием - технические, для разработчиков
        "formatting": {
            "missing_search_block": {
                "title": "⚠️ Ошибка форматирования: отсутствует SEARCH блок",
                "message": "В diff отсутствует блок с искомым текстом (<<<<<<< SEARCH).",
                "advice": "Пожалуйста, сгенерируйте diff заново с правильной структурой.",
                "internal": True
            },
            "missing_start_line": {
                "title": "⚠️ Ошибка форматирования: отсутствует :start_line:",
                "message": "В SEARCH блоке не указана начальная строка (:start_line:).",
                "advice": "Укажите номер строки, с которой начинается искомый фрагмент.",
                "internal": True
            },
            "missing_separator": {
                "title": "⚠️ Ошибка форматирования: отсутствует разделитель",
                "message": "Между SEARCH и REPLACE блоками отсутствует разделитель =======",
                "advice": "Пожалуйста, сгенерируйте diff заново с правильной структурой.",
                "internal": True
            },
            "empty_replace": {
                "title": "⚠️ Ошибка форматирования: пустой REPLACE блок",
                "message": "Блок для замены (REPLACE) пуст.",
                "advice": "Укажите новый код для замены или удалите блок, если хотите удалить код.",
                "internal": True
            },
            "wrong_search_marker": {
                "title": "⚠️ Ошибка форматирования: неправильный маркер поиска",
                "message": "Использован неправильный маркер начала SEARCH блока.",
                "advice": "Используйте точный маркер '<<<<<<< SEARCH'.",
                "internal": True
            },
            "wrong_replace_marker": {
                "title": "⚠️ Ошибка форматирования: неправильный маркер замены",
                "message": "Использован неправильный маркер конца REPLACE блока.",
                "advice": "Используйте точный маркер '>>>>>>> REPLACE'.",
                "internal": True
            },
            "wrong_separator": {
                "title": "⚠️ Ошибка форматирования: неправильный разделитель",
                "message": "Использован неправильный разделитель между блоками.",
                "advice": "Используйте точный разделитель '======='.",
                "internal": True
            },
            "duplicate_separators": {
                "title": "⚠️ Ошибка форматирования: дублирующиеся разделители",
                "message": "Обнаружены несколько разделителей подряд.",
                "advice": "Оставьте только один разделитель ======= между блоками.",
                "internal": True
            },
            "malformed_start_line": {
                "title": "⚠️ Ошибка форматирования: неправильный формат :start_line:",
                "message": "Директива :start_line: имеет неправильный формат.",
                "advice": "Используйте точный формат ':start_line:N', где N - номер строки.",
                "internal": True
            },
            "malformed_end_line": {
                "title": "⚠️ Ошибка форматирования: неправильный формат :end_line:",
                "message": "Директива :end_line: имеет неправильный формат.",
                "advice": "Используйте точный формат ':end_line:N', где N - номер строки.",
                "internal": True
            },
            "invalid_json": {
                "title": "⚠️ Ошибка форматирования: невалидный JSON",
                "message": "Аргументы tool call не являются валидным JSON.",
                "advice": "Проверьте синтаксис JSON в аргументах.",
                "internal": True
            },
            "invalid_arguments": {
                "title": "⚠️ Ошибка форматирования: неверный формат аргументов",
                "message": "Аргументы tool call должны быть объектом.",
                "advice": "Передавайте аргументы как JSON объект.",
                "internal": True
            },
            "missing_path": {
                "title": "⚠️ Ошибка форматирования: отсутствует путь к файлу",
                "message": "Не указан обязательный параметр 'path'.",
                "advice": "Укажите путь к файлу, который нужно изменить.",
                "internal": True
            },
            "missing_diff": {
                "title": "⚠️ Ошибка форматирования: отсутствует diff",
                "message": "Не указан обязательный параметр 'diff'.",
                "advice": "Предоставьте diff с изменениями.",
                "internal": True
            }
        },
        
        # Комбинированные сообщения для множественных ошибок
        "combined": {
            "multiple_content_issues": {
                "title": "❌ Обнаружено несколько проблем с содержимым",
                "message": "Найдено {count} проблем, требующих исправления.",
                "advice": "Пожалуйста, проверьте файл и скорректируйте запрос.",
                "priority": 1
            },
            "multiple_formatting_issues": {
                "title": "⚠️ Обнаружено несколько ошибок форматирования",
                "message": "Найдено {count} ошибок в структуре diff.",
                "advice": "Пожалуйста, сгенерируйте diff заново с правильным форматом.",
                "internal": True
            },
            "mixed_issues": {
                "title": "❌ Проблемы с содержимым и форматированием",
                "message": "Найдено {content_count} проблем с содержимым и {formatting_count} ошибок форматирования.",
                "advice": "Сначала исправьте содержимое, затем проверьте формат.",
                "priority": 1
            }
        }
    }
    
    def __init__(self, config_path: str = "config/tools.yaml"):
        """Инициализация процессора"""
        self._was_changed = False
        self.changes_log = []
        self._current_test_name = ""  # Для тестов
        self.workspace_path = None  # Будет установлен извне
        self.progress = {}
        # Менеджер повторных попыток
        self.retry_manager = RetryManager(max_retries=2)
        self.conversation_id = None  # Будет установлен извне
        
        # Загружаем конфигурацию инструментов
        self.tools_config = self._load_tools_config(config_path)
        
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
        default_config = {
            "apply_diff": True,
            "read_file": True,
            "ask_followup_question": True,
            "attempt_completion": True,
            "codebase_search": True,
            "execute_command": True,
            "function_replace": True,
            "list_files": True,
            "new_task": True,
            "read_command_output": True,
            "search_files": True,
            "switch_mode": False,
            "update_todo_list": True,
            "write_to_file": True,
            "mcp_context7_resolve_library_id": True,
            "mcp_context7_query_docs": True,
            "skill": False
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                tools_config = {}
                tools = config.get("tools")
                if tools and isinstance(tools, dict):
                    for tool_name, tool_enabled in tools.items():
                        tools_config[tool_name] = bool(tool_enabled)
                
                if tools_config:
                    return tools_config
                else:
                    return default_config
            else:
                return default_config
                
        except Exception as e:
            print(f"[-] Ошибка загрузки {config_path}: {str(e)}")
            return default_config
    
    def _parse_arguments(self, args_str: str, tool_name: str = "unknown", tool_call_id: str = "unknown") -> Dict[str, Any]:
        """Парсит строку аргументов в dict. Выбрасывает ArgumentParseError при ошибке."""
        if isinstance(args_str, dict):
            return args_str
        
        if not isinstance(args_str, str):
            raise ArgumentParseError(
                f"Неверный тип аргументов: ожидается str или dict, получен {type(args_str)}",
                tool_name=tool_name, tool_call_id=tool_call_id, original_args=str(args_str)
            )
        
        if not args_str.strip():
            raise ArgumentParseError(
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
            raise ArgumentParseError(
                f"JSON распарсен, но результат не dict (тип: {type(parsed).__name__})",
                tool_name=tool_name, tool_call_id=tool_call_id, original_args=args_str
            )
        except json.JSONDecodeError:
            pass
        
        # 2. Пробуем распарсить Python-подобный синтаксис
        try:
            return self._args_to_dict(args_str, tool_name, tool_call_id)
        except ArgumentParseError:
            pass
        
        raise ArgumentParseError(
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
            print(f"❌ function_replace: словарь file_lines пуст")
            return None
        
        if not isinstance(file_lines, dict):
            print(f"❌ function_replace: file_lines должен быть dict, получен {type(file_lines)}")
            return None
        
        # Собираем старый код из словаря в правильном порядке
        try:
            sorted_keys = sorted(file_lines.keys())
            old_code_lines = [file_lines[k] for k in sorted_keys]
            old_code = '\n'.join(old_code_lines)
        except Exception as e:
            print(f"❌ function_replace: ошибка при сборке кода: {e}")
            return None
        
        if not old_code.strip():
            print(f"❌ function_replace: нет содержимого функции '{function_name}' в {path}")
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
            
            # Получаем конфигурацию для инструмента
            tool_config = self.tools_config.get(tool_name, True)
            
            # Проверяем существование с учетом множественных определений
            exists = False
            
            if isinstance(tool_config, list):
                # Если это список - инструмент существует (множественные определения)
                # read_file: [false, true] -> exists = True
                exists = True
            else:
                exists = bool(tool_config)
            
            # Проверяем, есть ли инструмент в available_tools
            if tool_name not in available_tools:
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
            print(f"\n❌ НЕСУЩЕСТВУЮЩИЕ ИНСТРУМЕНТЫ:")
            print(f"   Всего: {len(invalid)}")
            for inv in invalid:
                print(f"   - {inv['tool_name']} (ID: {inv['tool_call_id']})")
                print(f"     Причина: {inv['reason']}")
                print(f"     Доступные инструменты: {inv['available_tools']}")
        
        return len(invalid) == 0, invalid

    def reset(self):
        """Сброс состояния"""
        self._was_changed = False
        self.changes_log = []

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
            template = self.RETRY_MESSAGES["content"].get(issue_type, {
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
            template = self.RETRY_MESSAGES["formatting"].get(issue_type, {
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
            template = self.RETRY_MESSAGES["content"][issue["type"]]
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
            template = self.RETRY_MESSAGES["formatting"][issue["type"]]
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
            template = self.RETRY_MESSAGES["combined"]["multiple_content_issues"]
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
            template = self.RETRY_MESSAGES["combined"]["multiple_formatting_issues"]
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
            template = self.RETRY_MESSAGES["combined"]["mixed_issues"]
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
            "max": self.retry_manager.max_retries,
            "remaining": self.retry_manager.max_retries - (attempt_count + 1)
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
            print(f"[-] Ошибка загрузки {config_path}: {e}")
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
        
        # 2. Исправляем parsed_args
        try:
            if tool_name == "read_file":
                if self._add_mode_slice_to_read_file(parsed_args):
                    changed = True
                    changes.append("read_file: добавлен mode=slice")
                if self._fix_read_file_offset(parsed_args):
                    changed = True
                    changes.append("read_file: offset исправлен с 0 на 1")
        
            if tool_name == "apply_diff":
                if self.fix_apply_diff(parsed_args):
                    changed = True
                    changes.append("apply_diff: конвертирован в формат RooCode")
            
            if tool_name == "ask_followup_question":
                if self._fix_followup_question(parsed_args):
                    changed = True
                    changes.append("ask_followup_question: follow_up строка -> массив")
            
            if self._fix_empty_path(parsed_args, tool_name):
                changed = True
                changes.append(f"{tool_name}: пустой path заменен на '.'")
            
            # 3а. Конвертация function_replace → apply_diff
            if tool_name == "function_replace":
                path = parsed_args.get("path", "")
                function_name = parsed_args.get("function", "")
                full_code = parsed_args.get("full_code", "")
                
                print(f"[DEBUG function_replace] Начало обработки")
                print(f"   - path: {path}")
                print(f"   - function_name: {function_name}")
                print(f"   - full_code length: {len(full_code)}")
                print(f"   - data is None: {data is None}")
                
                if data and data.get("type") == "function_content":
                    content_dict = data.get("content", {})
                    print(f"   - content_dict type: {type(content_dict)}")
                    print(f"   - content_dict keys count: {len(content_dict.keys()) if content_dict else 0}")
                    
                    if content_dict:
                        print(f"   ✓ Вызов _convert_function_replace...")
                        # ПЕРЕДАЕМ СЛОВАРЬ НАПРЯМУЮ
                        diff_result = self._convert_function_replace(path, function_name, full_code, content_dict)
                        
                        if diff_result:
                            print(f"   ✓ Конвертация успешна!")
                            tc['function']['name'] = 'apply_diff'
                            # Оставляем как dict, не сериализуем
                            tc['function']['arguments'] = {'path': path, 'diff': diff_result['diff']}
                            return True, [f"function_replace: конвертирован в apply_diff"]
                        else:
                            print(f"   ❌ _convert_function_replace вернул None")
                            return False, [f"function_replace: не удалось заменить '{function_name}'"]
                    else:
                        print(f"   ❌ content_dict пуст")
                        return False, [f"function_replace: нет данных функции '{function_name}' в {path}"]
                else:
                    print(f"   ❌ Нет данных function_content")
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

        # Логируем оригинал
        log_response(
            duration=answer.duration,
            status_code=answer.status_code
        )
        
        # Сохранение оригинального ответа теперь происходит в proxy.py (save_responce)
        
        # 1. Извлекаем tool calls из текста
        if self._extract_and_add_tool_calls(answer):
            self._was_changed = True
            self.changes_log.append("Извлечены tool calls из текста")
        
        # 3. Обрабатываем каждый tool call отдельно (модифицируем tc на месте)
        if answer.tool_calls:
            print(f"[DEBUG process] Step 2: processing tool calls")
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
                except ArgumentParseError as e:
                    # Логируем ошибку в консоль
                    print(f"\n❌ ОШИБКА ПАРСИНГА АРГУМЕНТОВ")
                    print(f"   Tool: {e.tool_name}")
                    print(f"   ID: {e.tool_call_id}")
                    print(f"   Ошибка: {e}")
                    print(f"   Аргументы: {e.original_args[:200] if e.original_args else 'None'}...")
                    
                    # Отправляем сообщение в LLM через существующий механизм retry
                    error_message = self._send_parse_error_to_llm(answer, e, tc['id'])
                    
                    # Возвращаем action для proxy.py
                    return {
                        'action': 'retry_with_llm',
                        'message': error_message,
                        'tool_call_id': tc['id']
                    }
                
                if self.progress[tc['id']] == "current":
                    
                    if (tool_name == "function_replace"):
                        func_path = tc_copy['function']['arguments']['path']
                        func_name = tc_copy['function']['arguments']['function']
                        if data and data.get("type") == "function_content" and data.get("path") == func_path:
                            print(f"[DEBUG process] Step 3: Use function '{func_name}()' content")
                            changed, changes = self.process_single_tool_call(tc_copy, i, data)
                            self.progress[tc['id']] = "completed"
                        else:
                            print(f"[DEBUG process] Step 3: Requesting function '{func_name}()' content from proxy.py")
                            return {
                                'action': 'request_function',
                                'path': func_path,
                                'function_name': func_name
                            }
                    else:
                        print(f"[DEBUG process] Step 2: processing tool_call[{i}] = {tool_name}")
                        changed, changes = self.process_single_tool_call(tc_copy, i)
                        print(f"   - tool name in tc_copy: {tc_copy['function']['name']}")
                        self.progress[tc['id']] = "completed"
                
                    # Сериализуем аргументы обратно в строку и обновляем tool_calls
                    tc_copy['function']['arguments'] = self._serialize_arguments(tc_copy['function']['arguments'])
                    answer.tool_calls[i] = copy.deepcopy(tc_copy)
                    
                    self.progress[tc['id']] = "completed"

                #if changed:
                #    self._was_changed = True
                #    self.changes_log.extend(changes)
            
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
            print(f"❌ Tool call validation failed: {validation_errors}")
            for error in validation_errors:
                error_tool_name = error.get('tool_name')
                error_message = error.get('message')
                print(f"   - {error_tool_name}: {error_message}")
        
        # 6. Логируем результат
        log_response(
            duration=answer.duration,
            status_code=200
        )
        
        # 7. Выводим статистику
        if self._was_changed:
            for change in self.changes_log:
                print(f"      - {change}")
        
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
            raise ArgumentParseError(
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
                raise ArgumentParseError(
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

    def fix_apply_diff(self, parsed_args: dict, debug: bool = False) -> bool:
        """
        Исправляет формат apply_diff.
        
        Args:
            parsed_args: Dict с аргументами (уже распарсенными)
            debug: Флаг отладки
            
        Returns:
            True если были изменения, иначе False
        """
        if "diff" not in parsed_args:
            return False
        
        old_diff = parsed_args["diff"]
        
        # Если формат уже валидный, но в REPLACE-блоке могут быть лишние пустые строки,
        # всё равно запускаем алгоритм для очистки
        new_diff = self._apply_diff_algorithm(old_diff, debug=debug)
        
        if new_diff and new_diff != old_diff:
            parsed_args["diff"] = new_diff
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
        
    def _apply_diff_algorithm(self, diff_text: str, debug: bool = False) -> Optional[str]:
        """
        Алгоритм исправления diff.
        
        Args:
            diff_text: исходный diff для исправления
            debug: если True, выводит отладочную информацию
        """
        if debug:
            print("\n" + "="*60)
            print("НАЧАЛЬНЫЙ DIFF:")
            print("*"*40)
            print(diff_text)
            print("*"*40)
        
        lines = diff_text.split('\n')
        changed = False
        step = 1
        
        def print_step(step_num, description):
            if debug:
                print(f"\n--- ШАГ {step_num}: {description} ---")
                print("*"*40)
                print('\n'.join(lines))
                print("*"*40)
        
        # Шаг 1: Ищем любые разделители, меняем на "======="
        separator_patterns = [
            # SEARCH маркеры (любые обрамления)
            r'^[<\[{]*\s*(?:SEARCH|SOURCE|SRC)\s*[>\]}]*$',
            
            # REPLACE маркеры (любые обрамления)  
            r'^[>\]}]*\s*REPLACE\s*[>\]}]*$',
            
            # Разделители (=== или ---)
            r'^[=\-]{3,}$',
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
            print_step(step, f"После замены разделителей на '=======' (найдено {len(separator_indices)} разделителей)")
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
            print_step(step, "После удаления дублирующихся разделителей")
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
            print_step(step, "После добавления разделителя в начало")
        step += 1
        
        # Шаг 4: Если в конце нет разделителя - добавляем
        if not separator_indices or separator_indices[-1] != len(lines) - 1:
            if debug:
                print("  Добавляем разделитель в конец")
            lines.append('=======')
            separator_indices.append(len(lines) - 1)
            changed = True
        
        if debug:
            print_step(step, "После добавления разделителя в конец")
        step += 1
        
        # Шаг 5: Проверяем количество разделителей
        if debug:
            print(f"  Текущее количество разделителей: {len(separator_indices)}")
        
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
            print_step(step, "После обработки случая с 3 разделителями")
        step += 1
        
        # Шаг 6: Проверяем количество разделителей
        if len(separator_indices) != 4:
            if debug:
                print(f"❌ Ошибка: найдено {len(separator_indices)} разделителей, нужно 4")
                print(f"   Индексы разделителей: {separator_indices}")
            return None
        
        if debug:
            print(f"  ✓ Найдено 4 разделителя на позициях: {separator_indices}")

        # Шаг 6.5: Убираем пробелы в начале строк с :start_line: и :end_line:
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(':start_line:') or stripped.startswith(':end_line:'):
                if line != stripped:
                    if debug:
                        print(f"  Убираем пробелы в строке {i}: '{line}' -> '{stripped}'")
                    lines[i] = stripped
                    changed = True

        # Шаг 7: Заменяем разделители по порядку на правильные
        # 1-й разделитель: <<<<<<< SEARCH
        old = lines[separator_indices[0]]
        lines[separator_indices[0]] = '<<<<<<< SEARCH'
        if debug:
            print(f"  Разделитель 1 (строка {separator_indices[0]}): '{old}' -> '<<<<<<< SEARCH'")
        
        # 2-й разделитель: -------
        old = lines[separator_indices[1]]
        lines[separator_indices[1]] = '-------'
        if debug:
            print(f"  Разделитель 2 (строка {separator_indices[1]}): '{old}' -> '-------'")
        
        # 3-й разделитель: =======
        old = lines[separator_indices[2]]
        lines[separator_indices[2]] = '======='
        if debug:
            print(f"  Разделитель 3 (строка {separator_indices[2]}): '{old}' -> '======='")
        
        # 4-й разделитель: >>>>>>> REPLACE
        old = lines[separator_indices[3]]
        lines[separator_indices[3]] = '>>>>>>> REPLACE'
        if debug:
            print(f"  Разделитель 4 (строка {separator_indices[3]}): '{old}' -> '>>>>>>> REPLACE'")
        
        if debug:
            print_step(step, "После замены разделителей на правильные")
        # Удаляем пустые строки в конце REPLACE блока
        # (между 3-м и 4-м разделителями)
        j = separator_indices[3] - 1
        while j > separator_indices[2] and not lines[j].strip():
            del lines[j]
            separator_indices[3] -= 1
            j -= 1
            changed = True


        result = '\n'.join(lines)

        
        if debug:
            print("\n" + "="*60)
            print("ИТОГОВЫЙ DIFF:")
            print("-"*40)
            print(result)
            print("="*60 + "\n")
        
        return result if changed else diff_text

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

    def _send_parse_error_to_llm(self, answer, error: ArgumentParseError, tool_call_id: str) -> Dict:
        """
        Формирует сообщение для отправки в LLM при ошибке парсинга аргументов.
        
        Args:
            answer: Объект Answer
            error: Исключение ArgumentParseError
            tool_call_id: ID tool call, который не удалось распарсить
            
        Returns:
            Dict с сообщением для добавления в историю
        """
        # Простое дружественное сообщение
        error_message = f"""❌ Не удалось обработать ваш tool call `{error.tool_name}`. Пожалуйста, попробуйте сформулировать запрос по-другому.

    **Проблема:** {str(error)}

    **Что делать:** Проверьте синтаксис аргументов и попробуйте еще раз.
    """
        
        # Логируем ошибку в консоль
        timestamp = datetime.now().isoformat()
        print(f"\n{'='*60}")
        print(f"❌ ОШИБКА ПАРСИНГА АРГУМЕНТОВ")
        print(f"{'='*60}")
        print(f"📅 Время: {timestamp}")
        print(f"🔧 Tool: {error.tool_name}")
        print(f"🆔 Tool Call ID: {error.tool_call_id}")
        print(f"📝 Ошибка: {error}")
        print(f"\n📄 Аргументы (первые 500 символов):")
        print(f"{'-'*40}")
        print(f"{error.original_args[:500] if error.original_args else 'None'}")
        if error.original_args and len(error.original_args) > 500:
            print(f"... (и еще {len(error.original_args) - 500} символов)")
        print(f"{'-'*40}")
        print(f"🤖 Модель: {answer.model}")
        print(f"💬 Conversation ID: {self.conversation_id or 'unknown'}")
        print(f"\n📤 Отправляем запрос на переформулирование в LLM...")
        print(f"{'='*60}\n")
        
        # Создаем сообщение для добавления в историю
        retry_message = {
            "role": "user",
            "content": error_message
        }
        
        return retry_message
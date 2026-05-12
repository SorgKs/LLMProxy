# requests.py
import json
import re
import copy
import os
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from scanner import ProjectStructureScanner


class RequestProcessor:
    """
    Класс для обработки и оптимизации запросов к LLM.
    Удаляет маркеры [200~ и [201~ из системных промптов и других полей.
    Добавляет функциональную структуру проекта в запросы.
    Оптимизирует описания инструментов (tools) для экономии токенов.
    Использует request.yaml для управления включением/исключением инструментов.
    """
    
    # Паттерны для поиска маркеров - ИСПРАВЛЕНО: используем raw строки и экранируем правильно
    MARKER_PATTERNS = [
        r'\x1b\[200~',   # Начальный маркер ESC [ 2 0 0 ~
        r'\x1b\[201~',   # Конечный маркер ESC [ 2 0 1 ~
    ]
    
    # Объединенный паттерн для поиска
    MARKER_REGEX = re.compile('|'.join(MARKER_PATTERNS))
    
    def __init__(self, config_path: str = "config/tools.yaml"):
        """
        Инициализация процессора запросов
        
        Args:
            workspace_path: Путь к рабочей директории проекта для сканирования структуры
            config_path: Путь к файлу конфигурации config/tools.yaml
        """
        self._was_changed = False
        self.changes_log = []
        
        # Статистика
        self.total_requests = 0
        self.modified_requests = 0
        self.total_chars_saved = 0
        self.total_percent_saved = 0
        self.model_stats = {}  # статистика по моделям
        
        # Фильтры для обработки контента
        self.filters: List[Callable] = [
            self._remove_markers,
            self._remove_system_prompt_noise,
            self._optimize_system_prompt
        ]
        
        # Путь к рабочей директории и сканер структуры проекта
        self.config_path = config_path
        
        # Загружаем конфигурацию инструментов из config/tools.yaml
        self.tools_config = self._load_tools_config(config_path)
    
    def build_correction_request(
        self, 
        original_body: Dict, 
        invalid_tool_calls: List[Dict], 
        available_tools: List[str]
    ) -> Dict:
        """
        Формирует НОВЫЙ запрос с сообщением об ошибке.
        Не отправляет, только создаёт тело запроса.
        """
        correction_body = copy.deepcopy(original_body)
        
        # Получаем индексы сообщений assistant с невалидными tool calls
        messages = correction_body.get("messages", [])
        indices_to_remove = []
        
        # Находим сообщения assistant с невалидными tool calls
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    # Проверяем, является ли этот tool call невалидным
                    for invalid in invalid_tool_calls:
                        if tc.get("id") == invalid.get("tool_call_id"):
                            indices_to_remove.append(i)
                            break
        
        # Удаляем найденные сообщения (в обратном порядке, чтобы не сбить индексы)
        for idx in sorted(indices_to_remove, reverse=True):
            del messages[idx]

        # Формируем сообщение об ошибке
        tool_names = [item["tool_name"] for item in invalid_tool_calls]
        tools_list = "\n".join([f"  - {tool}" for tool in available_tools])
        
        error_message = f"""❌ Error: The following tools do not exist: {', '.join(tool_names)}

    ✅ Available tools:
    {tools_list}

    Please use ONLY the tools listed above. Do not invent or call non-existent tools."""
        
        # Добавляем assistant сообщение с оригинальным tool call
        for item in invalid_tool_calls:
            correction_body["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [item["tool_call"]]
            })
            
            # Добавляем tool response с ошибкой
            correction_body["messages"].append({
                "role": "tool",
                "content": error_message,
                "tool_call_id": item["tool_call_id"]
            })
        
        return correction_body

    def _load_tools_from_folder(self, tools_folder: str = "tools") -> List[Dict]:
        """
        Динамически загружает все инструменты из папки tools/.
        
        Args:
            tools_folder: Путь к папке с инструментами
            
        Returns:
            List[Dict]: Список инструментов в формате OpenAI tools
        """
        tools = []
        
        if not os.path.exists(tools_folder):
            return tools
        
        try:
            for filename in sorted(os.listdir(tools_folder)):
                if filename.endswith(".json"):
                    filepath = os.path.join(tools_folder, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            tool_data = json.load(f)
                            if isinstance(tool_data, list):
                                for tool in tool_data:
                                    if self._validate_tool_format(tool):
                                        tools.append(tool)
                            elif self._validate_tool_format(tool_data):
                                tools.append(tool_data)
                    except Exception as e:
                        self.changes_log.append(f"⚠️ Ошибка загрузки {filename}: {str(e)}")
        except Exception as e:
            self.changes_log.append(f"⚠️ Ошибка чтения папки {tools_folder}: {str(e)}")
        
        return tools
    
    def _validate_tool_format(self, tool: Dict) -> bool:
        """
        Проверяет, что инструмент имеет правильный формат OpenAI tools.
        
        Args:
            tool: Инструмент для проверки
            
        Returns:
            True если формат корректный
        """
        if not isinstance(tool, dict):
            return False
        if tool.get("type") != "function":
            return False
        if "function" not in tool:
            return False
        func = tool["function"]
        if not isinstance(func, dict):
            return False
        if "name" not in func or "description" not in func:
            return False
        return True
    
    def _load_tools_config(self, config_path: str) -> Dict[str, bool]:
        """
        Загружает конфигурацию инструментов из config/tools.yaml
        и динамически загружает инструменты из папки tools/.
        
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
            "switch_mode": True,
            "update_todo_list": True,
            "write_to_file": True,
            "mcp_context7_resolve_library_id": True,
            "mcp_context7_query_docs": True,
            "skill": True
        }
        
        # Сначала загружаем инструменты из папки tools/
        tools_folder = os.path.join(os.path.dirname(config_path) or ".", "..", "tools")
        tools_folder = os.path.normpath(tools_folder)
        tools_from_folder = self._load_tools_from_folder(tools_folder)
        
        # Собираем имена инструментов из папки
        folder_tool_names = set()
        for tool in tools_from_folder:
            name = tool.get("function", {}).get("name", "")
            if name:
                folder_tool_names.add(name)
        
        # Если есть инструменты из папки, обновляем default_config
        if folder_tool_names:
            for name in folder_tool_names:
                if name not in default_config:
                    default_config[name] = True
            self.changes_log.append(f"📁 Загружено {len(tools_from_folder)} инструмент(ов) из папки tools/")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Парсим YAML структуру: ожидаем tools как корневой ключ
                tools_config = {}
                
                tools = config.get("tools")
                if tools and isinstance(tools, dict):
                    for tool_name, tool_enabled in tools.items():
                        tools_config[tool_name] = tool_enabled
                
                self.changes_log.append(f"📋 Загружена конфигурация инструментов из {config_path}")

                return tools_config
                
        except Exception as e:
            self.changes_log.append(f"❌ Ошибка загрузки {config_path}: {str(e)}")
            return default_config

    
    def _filter_tools_by_config(self, tools: List[Dict]) -> List[Dict]:
        """
        Фильтрует инструменты согласно конфигурации из config/tools.yaml
        Оставляет только инструменты со значением True
        Поддерживает список значений для множественного определения инструментов:
        [false, true] - первое вхождение удаляется, второе оставляется
        
        Args:
            tools: Исходный список инструментов
            
        Returns:
            List[Dict]: Отфильтрованный список инструментов
        """
        if not tools:
            return tools
        
        filtered_tools = []
        removed_tools = []
        
        # Счетчики для списков значений (для индексации вхождений)
        occurrence_counts: Dict[str, int] = {}
        
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            
            # Получаем конфигурацию для инструмента
            tool_config = self.tools_config.get(tool_name, True)

            # Поддержка списка значений для множественного определения
            if isinstance(tool_config, list):
                occurrence_counts[tool_name] = occurrence_counts.get(tool_name, 0) + 1
                index = occurrence_counts[tool_name] - 1
                print(f"[DEBUG] Tool: {tool_name}, config: {tool_config}, index: {index if isinstance(tool_config, list) else 'N/A'}")
                # Если индекс в пределах списка и значение True - оставляем
                if index < len(tool_config) and tool_config[index]:
                    filtered_tools.append(tool)
                else:
                    removed_tools.append(tool_name)
            elif tool_config:  # Обычное булево значение
                filtered_tools.append(tool)
            else:
                removed_tools.append(tool_name)
        
        if removed_tools:
            self.changes_log.append(f"🔧 Удалены инструменты: {', '.join(removed_tools)}")
            self._was_changed = True
        
        return filtered_tools
    
    @property
    def changed(self) -> bool:
        """Были ли изменения в последнем process()"""
        return self._was_changed
    
    def reset(self):
        """Сброс состояния"""
        self._was_changed = False
        self.changes_log = []

    def _remove_markers(self, text: str) -> str:
        """
        Удаляет маркеры [200~ и [201~ из текста.
        
        Args:
            text: Исходный текст
            
        Returns:
            Текст без маркеров
        """
        if not text:
            return text
        
        new_text = self.MARKER_REGEX.sub('', text)
        
        # Также удаляем возможные варианты с двойным экранированием
        new_text = new_text.replace('\\x1b\\[200~', '')
        new_text = new_text.replace('\\x1b\\[201~', '')
        new_text = new_text.replace('[200~', '')
        new_text = new_text.replace('[201~', '')
        
        if new_text != text:
            self.changes_log.append("Удалены маркеры [200~ и [201~")
        
        return new_text
    
    def _remove_system_prompt_noise(self, text: str) -> str:
        """
        Удаляет лишние символы из системных промптов.
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        if not text:
            return text
        
        # Удаляем множественные переносы строк
        new_text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Удаляем пробелы в начале и конце
        new_text = new_text.strip()
        
        # Удаляем специфичные артефакты
        new_text = re.sub(r'```\s*```', '', new_text)  # Пустые блоки кода
        
        if new_text != text:
            self.changes_log.append("Очищен системный промпт")
        
        return new_text
    
    def _optimize_system_prompt(self, text: str) -> str:
        """
        Фильтр для оптимизации структуры системного промпта.
        Применяет модификацию содержимого.
        
        Args:
            text: Исходный текст системного промпта
            
        Returns:
            Оптимизированный текст
        """
        return self._modify_system_prompt_content(text)
    
    def _modify_system_prompt_content(self, original_content: str) -> str:
        """
        Модифицирует содержимое системного промпта:
        - Удаляет текстовое описание из секции SYSTEM INFORMATION, оставляя только данные.
        - Оставляет все остальные секции без изменений.
        
        Args:
            original_content: Исходное содержимое системного промпта
            
        Returns:
            Модифицированное содержимое
        """
        if not original_content:
            return original_content
        
        # Паттерн для поиска всей секции SYSTEM INFORMATION
        # Ищем от "====\nSYSTEM INFORMATION" до следующего "====" или конца строки.
        # Флаг re.DOTALL позволяет точке совпадать с символом новой строки.
        pattern = r'(\n={4,}\nSYSTEM INFORMATION\n)(.*?)(\n={4,}\n|$)'
        
        def replacer(match):
            header = match.group(1)  # Начальный разделитель и заголовок
            body = match.group(2)    # Тело секции (текст + данные)
            footer_or_end = match.group(3) # Следующий разделитель или конец
            
            # Извлекаем только строки с данными (содержат ':')
            data_lines = []
            for line in body.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('#'): # Простой эвристический фильтр
                    # Оставляем только ключ: значение, без лишнего текста
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        data_lines.append(f"{key}: {value}")
            
            # Формируем новое тело секции только из данных
            if data_lines:
                new_body = "\n" + "\n".join(data_lines) + "\n"
            else:
                new_body = "\n"  # Пустая строка, если данных нет
            
            # Возвращаем заголовок + новые данные + то, что было после (разделитель или конец)
            return header + new_body + footer_or_end
        
        # Выполняем замену
        modified_content, number_of_subs = re.subn(pattern, replacer, original_content, flags=re.DOTALL)
        
        if number_of_subs > 0:
            self.changes_log.append("🔄 Системный промпт оптимизирован (удалено описание SYSTEM INFORMATION)")
            return modified_content
        else:
            # Если секция не найдена, возвращаем оригинал
            return original_content
    
    def _process_messages(self, messages: List[Dict]) -> tuple:
        """
        Обрабатывает все сообщения в запросе.
        
        Args:
            messages: Список сообщений
            
        Returns:
            Tuple[bool, int, int] - (были_изменения, исходная_длина, новая_длина)
        """
        changed = False
        original_total = 0
        new_total = 0
        
        for i, message in enumerate(messages):
            if "content" in message and isinstance(message["content"], str):
                content = message["content"]
                original_total += len(content)
                
                # Применяем все фильтры
                for filter_func in self.filters:
                    new_content = filter_func(content)
                    if new_content != content:
                        changed = True
                        self.changes_log.append(f"message[{i}]: {filter_func.__name__}")
                        content = new_content
                
                message["content"] = content
                new_total += len(content)
        
        return changed, original_total, new_total
    
    def _process_tools(self, tools: List[Dict]) -> tuple:
        """
        Обрабатывает инструменты в запросе.
        
        Args:
            tools: Список инструментов
            
        Returns:
            tuple: (были_ли_изменения, отфильтрованный_список_инструментов)
        """
        changed = False
        
        # Сначала фильтруем инструменты согласно конфигурации
        filtered_tools = self._filter_tools_by_config(tools)
        
        # Проверяем, изменился ли список
        if len(filtered_tools) != len(tools):
            changed = True
        
        # Применяем оптимизации к оставшимся инструментам
        for tool in filtered_tools:
            # Применяем общие фильтры
            if "function" in tool and "description" in tool["function"]:
                desc = tool["function"]["description"]
                if isinstance(desc, str):
                    new_desc = self._remove_markers(desc)
                    if new_desc != desc:
                        tool["function"]["description"] = new_desc
                        changed = True
                        self.changes_log.append("tool description: удалены маркеры")
            
            # Применяем специализированную обработку для read_file (только если он включен)
            if self.tools_config.get("read_file", True):
                if self._process_read_file_tool(tool):
                    changed = True
        
        return changed, filtered_tools
    
    def _process_read_file_tool(self, tool: Dict) -> bool:
        """
        Специализированная обработка для tool 'read_file':
        - Упрощает description до "returns up to 2000 lines per file"
        - В описании параметра path оставляет только "relative to the workspace"
        - Удаляет параметр mode полностью
        - Удаляет все описания касающиеся slice/indentation
        - Удаляет блок indentation полностью
        
        Args:
            tool: Инструмент для обработки
            
        Returns:
            bool: Были ли изменения
        """
        changed = False
        
        # Проверяем, что это нужный нам tool
        if not (tool.get("type") == "function" and 
                tool.get("function", {}).get("name") == "read_file"):
            return False
        
        function = tool.get("function", {})
        
        # 1. Упрощаем основное description
        if "description" in function:
            original_desc = function["description"]
            # Оставляем только основную фразу про 2000 строк
            new_desc = "Read a file from the local filesystem. returns up to 2000 lines per file"
            if new_desc != original_desc:
                function["description"] = new_desc
                changed = True
                self.changes_log.append("🔧 read_file: description упрощен")
        
        # 2. Обрабатываем параметры
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        
        # 2.1. Упрощаем description для path
        if "path" in properties and "description" in properties["path"]:
            original_desc = properties["path"]["description"]
            new_desc = "relative to the workspace"
            if new_desc != original_desc:
                properties["path"]["description"] = new_desc
                changed = True
                self.changes_log.append("🔧 read_file.path: description упрощен")
        
        # 2.2. Удаляем параметр mode полностью
        if "mode" in properties:
            del properties["mode"]
            # Также удаляем mode из required, если он там есть
            if "required" in parameters and "mode" in parameters["required"]:
                parameters["required"].remove("mode")
            changed = True
            self.changes_log.append("🔧 read_file: параметр mode удален")
        
        # 2.3. Удаляем параметр offset (он тоже связан со slice)
        #if "offset" in properties:
        #    del properties["offset"]
        #    if "required" in parameters and "offset" in parameters["required"]:
        #        parameters["required"].remove("offset")
        #    changed = True
        #    self.changes_log.append("🔧 read_file: параметр offset удален")
        
        # 2.4. Удаляем параметр limit (он тоже связан со slice)
        #if "limit" in properties:
        #    del properties["limit"]
        #    if "required" in parameters and "limit" in parameters["required"]:
        #        parameters["required"].remove("limit")
        #    changed = True
        #    self.changes_log.append("🔧 read_file: параметр limit удален")
        
        # 2.5. Удаляем все упоминания slice/indentation из описаний
        slice_terms = ['slice', 'indent', 'indentation', 'slicing', 'line range', 'partial read']
        
        for param_name, param_data in properties.items():
            if "description" in param_data:
                original_desc = param_data["description"]
                new_desc = original_desc
                
                # Удаляем фразы с slice/indentation
                for term in slice_terms:
                    # Паттерны: "using slice", "with slice", "slice:", "indentation", и т.д.
                    patterns = [
                        rf'(?i)(?:using|with|by|in)?\s*{term}\s*(?::|,|\.|;|$)?',
                        rf'(?i){term}\s+(?:mode|parameter|option|method|approach)',
                        rf'(?i)(?:line\s+)?{term}\s+(?:range|start|end|based)',
                        rf'(?i)supports?\s+{term}',
                        rf'(?i)for\s+{term}\s+reads?'
                    ]
                    for pattern in patterns:
                        new_desc = re.sub(pattern, '', new_desc, flags=re.IGNORECASE)
                
                # Удаляем фразы про offset/limit
                offset_patterns = [
                    r'(?i)(?:with|using|by)?\s*offset\s*(?::|,|\.|;|$)?',
                    r'(?i)(?:with|using|by)?\s*limit\s*(?::|,|\.|;|$)?',
                    r'(?i)(?:start|end)(?:\s+line)?\s*(?::|,|\.|;)?',
                    r'(?i)line\s+(?:number|range|index)'
                ]
                for pattern in offset_patterns:
                    new_desc = re.sub(pattern, '', new_desc, flags=re.IGNORECASE)
                
                # Очищаем от лишних пробелов и знаков препинания
                new_desc = re.sub(r'\s+', ' ', new_desc).strip()
                new_desc = re.sub(r',\s*,', ',', new_desc)
                new_desc = re.sub(r'\.\s*\.', '.', new_desc)
                new_desc = re.sub(r';\s*;', ';', new_desc)
                new_desc = re.sub(r'^\s*[,.;]|\s*[,.;]\s*$', '', new_desc)
                
                # Если после очистки описание стало пустым, ставим заглушку
                if not new_desc:
                    if param_name == "path":
                        new_desc = "relative to the workspace"
                    else:
                        new_desc = f"Parameter: {param_name}"
                
                if not new_desc.endswith('.'):
                    new_desc += '.'
                
                if new_desc != original_desc:
                    param_data["description"] = new_desc
                    changed = True
                    self.changes_log.append(f"🔧 read_file.{param_name}: очищено от slice/indentation")
        
        # 3. Удаляем блок indentation полностью, если он есть
        if "indentation" in function:
            del function["indentation"]
            changed = True
            self.changes_log.append("🔧 read_file: блок indentation удален")
        
        # 4. Проверяем наличие других полей с indentation в разных местах
        if "indentation" in parameters:
            del parameters["indentation"]
            changed = True
            self.changes_log.append("🔧 read_file: поле indentation в parameters удалено")
        
        if "indentation" in properties:
            del properties["indentation"]
            if "required" in parameters and "indentation" in parameters["required"]:
                parameters["required"].remove("indentation")
            changed = True
            self.changes_log.append("🔧 read_file: параметр indentation удален")
        
        # 5. Проверяем, не осталось ли пустых required
        if "required" in parameters and not parameters["required"]:
            del parameters["required"]
            changed = True
            self.changes_log.append("🔧 read_file: удален пустой required")
        
        return changed
    
    def process(self, body: dict):
        """
        Обрабатывает тело запроса: логирует, модифицирует, удаляет системные промпты,
        добавляет структуру проекта, оптимизирует описания инструментов.
        
        Args:
            body: Тело запроса
            method: HTTP метод
            url: URL запроса
            headers: Заголовки запроса
            
        Returns:
            Модифицированное тело запроса
        """
        
        self.reset()
    
        # Сохраняем исходный запрос в requests/
        self._save_original_request(body)
        
        # Делаем глубокую копию, чтобы не изменять оригинал
        modified_body = copy.deepcopy(body)
        modifications = []
        
        # Получаем модель для статистики
        model = modified_body.get("model", "unknown")
        
        # Обновляем общую статистику
        self.total_requests += 1
        
        # Инициализируем статистику для модели, если нужно
        if model not in self.model_stats:
            self.model_stats[model] = {
                "requests": 0,
                "modified": 0,
                "chars_saved": 0
            }
        
        self.model_stats[model]["requests"] += 1

        # Обрабатываем messages (удаление маркеров, оптимизация промптов)
        if "messages" in modified_body:
            messages_changed, original_len, new_len = self._process_messages(modified_body["messages"])
            
            if messages_changed:
                self._was_changed = True
                self.modified_requests += 1
                self.model_stats[model]["modified"] += 1
                
                chars_saved = original_len - new_len
                self.total_chars_saved += chars_saved
                self.model_stats[model]["chars_saved"] += chars_saved
                
                if original_len > 0:
                    percent_saved = (chars_saved / original_len) * 100
                else:
                    percent_saved = 0
                
                # Обновляем средний процент
                if self.modified_requests > 0:
                    self.total_percent_saved = (
                        (self.total_percent_saved * (self.modified_requests - 1) + percent_saved) 
                        / self.modified_requests
                    )
                
                modifications.append(f"messages: сохранено {chars_saved} символов ({percent_saved:.1f}%)")
        
        # Обрабатываем tools (фильтрация по конфигурации и оптимизация)
        if "tools" in modified_body:
            tools_changed, filtered_tools = self._process_tools(modified_body["tools"])
            if tools_changed:
                self._was_changed = True
                modified_body["tools"] = filtered_tools
                modifications.append("tools: оптимизированы описания и отфильтрованы по конфигурации")

        # Добавляем инструменты из папки tools/ (если они включены в конфигурации)
        # Собираем инструменты из folder/ и фильтруем по конфигурации
        enabled_folder_tools = []
        for tool in self._load_tools_from_folder(os.path.join(os.path.dirname(self.config_path) if self.config_path else "config", "..", "tools")):
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name and self.tools_config.get(tool_name, True):
                # Проверяем, нет ли уже этого инструмента в запросе
                existing_names = {t.get("function", {}).get("name", "") for t in modified_body.get("tools", [])}
                if tool_name not in existing_names:
                    enabled_folder_tools.append(tool)
        
        if enabled_folder_tools:
            if "tools" not in modified_body:
                modified_body["tools"] = []
            modified_body["tools"].extend(enabled_folder_tools)
            self._was_changed = True
            tool_names = ", ".join(t.get("function", {}).get("name", "") for t in enabled_folder_tools)
            modifications.append(f"tools: добавлены инструменты из папки ({tool_names})")

        # Логируем модифицированный запрос, если были изменения
        if self._was_changed:
            # Сохраняем модифицированный запрос в requests/
            self._save_modified_request(modified_body, modifications)
            
            # Добавляем в общий лог изменений
            self.changes_log.extend(modifications)
        
        return modified_body

    def _detect_failed_apply_diff(self, messages: List[Dict]) -> None:
        """
        Анализирует сообщения на наличие неудачного apply_diff.
        Выводит сообщение в консоль при обнаружении.
        """
        # Паттерны для поиска сообщений о неудачном apply_diff
        FAILED_APPLY_PATTERNS = [
            r'apply_diff.*не удалось',
            r'apply_diff.*failed',
            r'apply_diff.*error',
            r'Failed to apply.*diff',
            r'Ошибка применения.*apply_diff',
            r'Content mismatch.*apply_diff',
            r'Search block.*not found',
            r'не удалось применить изменения',
            r'apply_diff.*не применён',
        ]
        
        combined_pattern = re.compile('|'.join(FAILED_APPLY_PATTERNS), re.IGNORECASE)
        
        for i, message in enumerate(messages):
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                if isinstance(content, str) and combined_pattern.search(content):
                    print(f"\n⚠️ UNSUCCESSFUL apply_diff DETECTED in client request")
                    print(f"   Message {i}: {message.get('role', 'unknown')}")
                    preview = content[:300].replace('\n', ' ')
                    print(f"   Content preview: {preview}...")
                    print()
                    break

    def _save_original_request(self, body: dict) -> None:
        """Сохраняет исходный запрос в файл requests/original_request_<timestamp>.json"""
        try:
            os.makedirs('requests', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"requests/original_request_{timestamp}.json"
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "body": body
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[-] Ошибка при сохранении исходного запроса: {e}")

    def _save_modified_request(self, modified_body: dict, modifications: List[str]) -> None:
        """Сохраняет модифицированный запрос в файл requests/modified_request_<timestamp>.json"""
        try:
            os.makedirs('requests', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"requests/modified_request_{timestamp}.json"
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "body": modified_body,
                "modifications": modifications
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[-] Ошибка при сохранении модифицированного запроса: {e}")
    
    def get_summary_stats(self) -> Dict:
        """
        Возвращает сводную статистику по обработке запросов.
        
        Returns:
            Dict со статистикой
        """
        avg_saved = self.total_percent_saved if self.modified_requests > 0 else 0
        
        # Оценка токенов (грубо: 1 токен ≈ 4 символа)
        tokens_saved = self.total_chars_saved // 4
        
        return {
            "total_requests": self.total_requests,
            "modified_requests": self.modified_requests,
            "modification_percent": (self.modified_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "total_chars_saved": self.total_chars_saved,
            "avg_percent_saved": avg_saved,
            "estimated_tokens_saved": tokens_saved,
            "by_model": self.model_stats
        }
    
    def print_summary(self):
        """Выводит красиво отформатированную статистику"""
        stats = self.get_summary_stats()
        
        print("\n" + "="*80)
        print("📊 СТАТИСТИКА ОБРАБОТКИ ЗАПРОСОВ")
        print("="*80)
        print(f"Всего запросов: {stats['total_requests']}")
        print(f"Модифицировано: {stats['modified_requests']} ({stats['modification_percent']:.1f}%)")
        print(f"Всего сохранено символов: {stats['total_chars_saved']}")
        print(f"Средняя экономия: {stats['avg_percent_saved']:.1f}%")
        print(f"Примерно сохранено токенов: {stats['estimated_tokens_saved']}")
        print("-"*80)
        
        if stats['by_model']:
            print("По моделям:")
            for model, model_stats in stats['by_model'].items():
                mod_percent = (model_stats['modified'] / model_stats['requests'] * 100) if model_stats['requests'] > 0 else 0
                print(f"  • {model}:")
                print(f"      Запросов: {model_stats['requests']}")
                print(f"      Модифицировано: {model_stats['modified']} ({mod_percent:.1f}%)")
                print(f"      Сохранено символов: {model_stats['chars_saved']}")
        
        # Добавляем информацию о кэше структуры
        if self._cached_structure:
            print("-"*80)
            print(f"📁 Структура проекта закэширована: {len(self._cached_structure)} символов")
        
        # Добавляем информацию о конфигурации инструментов
        print("-"*80)
        print("🔧 Конфигурация инструментов из config/tools.yaml:")
        for tool_name, enabled in self.tools_config.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {tool_name}")
        
        print("="*80)


# Для обратной совместимости
def process_request(body: dict, workspace_path: Optional[str] = None, config_path: str = "request.yaml") -> dict:
    """
    Упрощенная функция для обработки запроса.
    
    Args:
        body: Тело запроса
        workspace_path: Путь к рабочей директории для сканирования структуры
        config_path: Путь к файлу конфигурации request.yaml
        
    Returns:
        Модифицированное тело запроса
    """
    processor = RequestProcessor(workspace_path=workspace_path, config_path=config_path)
    return processor.process(body)


def clean_system_prompt(text: str) -> str:
    """
    Очищает системный промпт от маркеров.
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    processor = RequestProcessor()
    return processor._remove_markers(text)
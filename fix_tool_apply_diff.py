# apply_diff_fixer.py
"""
Класс для исправления и валидации apply_diff tool calls.
"""
import re
import logging
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import handlers

logger = logging.getLogger(__name__)

# Директория для логов ошибок
ERROR_LOG_DIR = "logs"
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

# Ограничения для файлов логов
MAX_LOG_FILES = 20
MAX_LOG_AGE_HOURS = 24


def _cleanup_old_log_files():
    """
    Удаляет старые лог-файлы ошибок.
    Оставляет не более MAX_LOG_FILES файлов и не старше MAX_LOG_AGE_HOURS часов.
    """
    try:
        pattern = os.path.join(ERROR_LOG_DIR, "apply_diff_error_*.log")
        files = glob.glob(pattern)
        
        if not files:
            return
        
        # Сортируем по времени модификации (новые первые)
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Удаляем файлы сверх лимита
        if len(files) > MAX_LOG_FILES:
            for f in files[MAX_LOG_FILES:]:
                try:
                    os.remove(f)
                except OSError:
                    pass
        
        # Удаляем файлы старше MAX_LOG_AGE_HOURS
        cutoff_time = datetime.now() - timedelta(hours=MAX_LOG_AGE_HOURS)
        for f in files[:MAX_LOG_FILES]:
            try:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(f))
                if file_mtime < cutoff_time:
                    os.remove(f)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"Ошибка при очистке лог-файлов: {e}")


def _dump_log_to_file(log_buffer: List[str], prefix: str = "apply_diff_error"):
    """
    Сбрасывает содержимое лог-буфера в файл с datetime в имени.
    
    Args:
        log_buffer: Буфер лога для записи
        prefix: Префикс имени файла
    """
    if not log_buffer:
        return
    
    try:
        # Очищаем старые файлы перед созданием нового
        _cleanup_old_log_files()
        
        # Формируем имя файла с datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.log"
        filepath = os.path.join(ERROR_LOG_DIR, filename)
        
        # Записываем содержимое буфера в файл
        with open(filepath, "w", encoding="utf-8") as f:
            f.write('\n'.join(log_buffer))
            f.write('\n')
        
        logger.info(f"Лог ошибок записан в файл: {filepath}")
    except Exception as e:
        logger.error(f"Ошибка при записи лога в файл: {e}")


class FixToolApplyDiff:
    """
    Класс для комплексного исправления apply_diff:
    форматирование и поиск start_line.
    """
    
    def __init__(self):
        self._was_changed = False
        self._log_buffer: List[str] = []
    
    @property
    def changed(self) -> bool:
        """Были ли изменения"""
        return self._was_changed
    
    @property
    def log(self) -> str:
        """Возвращает содержимое лог-буфера как строку"""
        return '\n'.join(self._log_buffer)
    
    @property
    def log_lines(self) -> List[str]:
        """Возвращает копию лог-буфера как список строк"""
        return self._log_buffer.copy()
    
    def _log(self, message: str, level: str = "DEBUG"):
        """
        Внутренний метод логирования.
        Пишет в буфер и в logger (для всех уровней).
        
        Args:
            message: Сообщение для логирования
            level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        # Добавляем в буфер
        self._log_buffer.append(formatted_message)
        
        # Также пишем в logger в зависимости от уровня
        level_upper = level.upper()
        if level_upper == "DEBUG":
            logger.debug(message)
        elif level_upper == "INFO":
            logger.info(message)
        elif level_upper == "WARNING":
            logger.warning(message)
        elif level_upper == "ERROR":
            logger.error(message)
        else:
            logger.debug(message)
    
    def _log_colored(self, message: str, color: str = "", level: str = "DEBUG"):
        """
        Логирование с цветным форматированием (для обратной совместимости).
        Цвет игнорируется в буфере, но используется для консоли через logger.
        
        Args:
            message: Сообщение для логирования
            color: Цветовой код (игнорируется в буфере)
            level: Уровень логирования
        """
        self._log(message, level)
    
    def _dump_log_on_error(self):
        """
        Сбрасывает лог в файл при ошибке.
        """
        if self._log_buffer:
            _dump_log_to_file(self._log_buffer)
    
    def _print_diff(self, lines: List[str], title: str) -> None:
        """Выводит preview diff после каждого шага"""
        self._log(f"\n{title}")
        self._log("=" * 60)
        for i, line in enumerate(lines):
            self._log(line)
    
    def _deep_search(
        self, 
        diff: str, 
        file_content: Dict[int, str], 
        debug: bool = False
    ) -> Tuple[Optional[str], List[str]]:
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
            self._log("\n" + "=" * 60)
            self._log("Deep Search: Анализ проблемного diff")
            self._log("=" * 60)
            self._print_diff(lines, "Исходный diff")
        
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
                        self._log(f"✓ Стартовый маркер: {start_marker[:50]}")
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
                    self._log(f"✓ Маркер строки: {line_marker[:50]}")
                
                # 1.3 Поиск разделителя после маркера строки (строго следующая строка)
                if remaining_lines:
                    next_next_line = remaining_lines[0].strip()
                    for pattern in separator_patterns:
                        if re.match(pattern, next_next_line):
                            separator_after_line = remaining_lines.pop(0)
                            issues.append(f"Найден разделитель после маркера: {separator_after_line}")
                            if debug:
                                self._log(f"✓ Разделитель после маркера: {separator_after_line}")
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
                        self._log(f"✓ Финальный маркер: {end_marker[:50]}")
                    break
        
        # 1.5 Оставшиеся строки = чистый контент
        clean_lines = remaining_lines
        clean_content = '\n'.join(clean_lines)
        
        if debug:
            self._log("\nЧистый контент (без маркеров):")
            self._log("-" * 40)
            for i, line in enumerate(clean_lines[:10]):
                repr_line = line.replace(' ', '·').replace('\t', '→')
                self._log(f"{i:3}: '{repr_line[:80]}'")
            if len(clean_lines) > 10:
                self._log(f"... и еще {len(clean_lines) - 10} строк")
            self._log("-" * 40)
        
        # Фаза 2: Поиск в файле
        if not file_content:
            issues.append("Нет содержимого файла для верификации")
            self._log("Нет содержимого файла для верификации", "ERROR")
            return None, issues
        
        # Берем первые 3 строки из чистого контента (как есть, включая пустые, сохраняя пробелы)
        search_prefix = clean_lines[:3]
        
        if not search_prefix:
            issues.append("Чистый контент пуст")
            self._log("Чистый контент пуст", "ERROR")
            return None, issues
        
        if debug:
            self._log("\nПоиск в файле первых 3 строк SEARCH блока:")
            for i, line in enumerate(search_prefix):
                repr_line = line.replace(' ', '·').replace('\t', '→')
                self._log(f"  {i+1}: '{repr_line[:80]}'")
        
        # Ищем эти строки в файле
        file_lines_list = [file_content[k] for k in sorted(file_content.keys())]
        found_position = None
        whitespace_warning = False
        
        for i in range(len(file_lines_list) - len(search_prefix) + 1):
            match = True
            trim_match = True
            
            for j, search_line in enumerate(search_prefix):
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
                    self._log(f"✓ Найдено ТОЧНОЕ совпадение на позиции {i} (строка {i+1} в файле)")
                break
            
            # Если точного нет, но trim совпал - запоминаем для предупреждения
            if trim_match and not match:
                whitespace_warning = True
                if debug:
                    self._log(f"⚠ На позиции {i}: trim версии совпадают, но есть различия в пробелах")
                    for j, search_line in enumerate(search_prefix[:2]):
                        if i + j < len(file_lines_list):
                            file_line = file_lines_list[i + j]
                            self._log(f"    Файл:   '{file_line.replace(' ', '·').replace('\t', '→')[:60]}'")
                            self._log(f"    Diff:   '{search_line.replace(' ', '·').replace('\t', '→')[:60]}'")
        
        if found_position is None:
            issues.append("Не удалось найти начало SEARCH блока в файле")
            if whitespace_warning:
                issues.append("ПРЕДУПРЕЖДЕНИЕ: Найдены совпадения trim-версий, но различия в пробелах")
            self._log("Не удалось найти начало SEARCH блока в файле", "ERROR")
            return None, issues
        
        # Фаза 3: Поиск истинного разделителя
        file_idx = found_position
        content_idx = 0
        true_separator = None
        true_separator_pos = None
        
        if debug:
            self._log("\nПоиск расхождения:")
        
        while file_idx < len(file_lines_list) and content_idx < len(clean_lines):
            file_line = file_lines_list[file_idx]
            content_line = clean_lines[content_idx]
            
            if debug:
                file_repr = file_line.replace(' ', '·').replace('\t', '→')[:50]
                content_repr = content_line.replace(' ', '·').replace('\t', '→')[:50]
                self._log(f"  Сравнение: файл[{file_idx}]='{file_repr}' vs контент[{content_idx}]='{content_repr}'")
            
            if file_line != content_line:
                if debug:
                    self._log(f"⚠ Расхождение на позиции {content_idx}")
                
                if content_idx < len(clean_lines):
                    potential_separator = clean_lines[content_idx].strip()
                    for pattern in separator_patterns:
                        if re.match(pattern, potential_separator):
                            true_separator = potential_separator
                            true_separator_pos = content_idx
                            issues.append(f"Найден истинный разделитель: {true_separator} на позиции {true_separator_pos}")
                            if debug:
                                self._log(f"✓ Истинный разделитель: {true_separator}")
                            break
                
                if true_separator is None:
                    issues.append(f"ОШИБКА: В месте расхождения нет разделителя (строка: '{clean_lines[content_idx][:50]}')")
                    self._log(f"ОШИБКА: В месте расхождения нет разделителя (строка: '{clean_lines[content_idx][:50]}')", "ERROR")
                    return None, issues
                
                break
            
            file_idx += 1
            content_idx += 1
        
        if true_separator is None:
            issues.append("Не удалось найти истинный разделитель (достигнут конец файла или контента)")
            self._log("Не удалось найти истинный разделитель (достигнут конец файла или контента)", "ERROR")
            return None, issues
        
        # Фаза 4: Пересборка diff с экранированием ложных маркеров
        search_content_lines = clean_lines[:true_separator_pos]
        replace_content_lines = clean_lines[true_separator_pos + 1:] if true_separator_pos + 1 < len(clean_lines) else []
        
        if debug:
            self._log("\nРазделение контента:")
            self._log(f"  SEARCH блок: {len(search_content_lines)} строк")
            self._log(f"  REPLACE блок: {len(replace_content_lines)} строк")
            self._log(f"  Истинный разделитель: '{true_separator}'")
        
        def escape_fake_markers(lines_list: List[str]) -> List[str]:
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
        
        final_lines = []
        final_lines.append('<<<<<<< SEARCH')
        
        actual_line_num = found_position + 1
        if line_number:
            final_lines.append(f':start_line:{line_number}')
        else:
            final_lines.append(f':start_line:{actual_line_num}')
        
        final_lines.append('-------')
        final_lines.extend(search_content_escaped)
        final_lines.append('=======')
        final_lines.extend(replace_content_escaped)
        final_lines.append('>>>>>>> REPLACE')
        
        final_diff = '\n'.join(final_lines)
        
        if debug:
            self._log("\nФинальный diff после deep_search:")
            self._print_diff(final_lines, "Deep Search Result")
            self._log("\nПроблемы и исправления:")
            for issue in issues:
                self._log(f"  • {issue}")
        
        return final_diff, issues
    
    def _find_search_block_line(
        self, 
        file_content: Dict[int, str], 
        lines: List[str], 
        search_block: tuple
    ) -> Optional[int]:
        """
        Находит номер строки, где начинается SEARCH блок в файле.
        
        Args:
            file_content: Словарь {номер_строки: содержимое}
            lines: Список строк diff (уже разбитый на строки)
            search_block: Кортеж (start_line, length) - начало и длина SEARCH блока в diff
            
        Returns:
            Номер строки в файле (1-based) или None если не найден
        """
        if not all(isinstance(k, int) for k in file_content.keys()):
            logger.error(f"_find_search_block_line: ключи file_content должны быть int")
            return None
        
        if not file_content or not lines or not search_block:
            return None
            
        if isinstance(search_block, (list, tuple)) and len(search_block) == 2:
            start_line, block_length = search_block
        else:
            return None
        
        if start_line < 0 or start_line >= len(lines):
            return None
        
        sorted_keys = sorted(file_content.keys())
        file_lines = [file_content[k] for k in sorted_keys]
        line_nums = sorted_keys
        
        char_replacements = {
            '"': '«',
            "'": '′',
            '-': '—',
            '«': '"',
            '′': "'",
            '—': '-'
        }
        
        def normalize_line(line: str) -> str:
            if line.startswith('| '):
                line = line[2:]
            return line.strip()
        
        for i in range(len(file_lines) - block_length + 1):
            match = True
            
            for j in range(block_length):
                if i + j >= len(file_lines):
                    match = False
                    break
                
                file_line = file_lines[i + j]
                search_line = lines[start_line + j]
                
                file_line_normalized = file_line.strip()
                search_line_normalized = normalize_line(search_line)
                
                if file_line_normalized == '' and search_line_normalized == '':
                    continue
                
                if file_line_normalized == search_line_normalized:
                    continue
                
                attempt = 0
                max_attempts = 10
                current_line = search_line_normalized
                
                while attempt < max_attempts and file_line_normalized != current_line:
                    min_len = min(len(file_line_normalized), len(current_line))
                    mismatch_pos = None
                    for pos in range(min_len):
                        if file_line_normalized[pos] != current_line[pos]:
                            mismatch_pos = pos
                            break
                    
                    if mismatch_pos is None:
                        if len(file_line_normalized) != len(current_line):
                            break
                    
                    diff_char = current_line[mismatch_pos]
                    file_char = file_line_normalized[mismatch_pos]
                    
                    replaced = False
                    for s1, s2 in char_replacements.items():
                        if diff_char == s1 and file_char == s2:
                            line_list = list(lines[start_line + j])
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
                        break
                    
                    attempt += 1
                
                if file_line_normalized != current_line:
                    match = False
                    break
            
            if match:
                result_line = line_nums[i]
                for j in range(block_length):
                    original_file_line = file_lines[i + j]
                    lines[start_line + j] = original_file_line
                    self._was_changed = True
                
                return result_line
        
        return None
    
    def fix(
        self, 
        tc: dict, 
        data: dict, 
        debug: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Комплексное исправление apply_diff: форматирование и поиск start_line.
        
        Args:
            tc: Словарь с аргументами tool call (будет модифицирован)
            data: Данные файла {"type": "file_content", "content": {...}}
            debug: Режим отладки
            
        Returns:
            Tuple[bool, Optional[str]]: (были ли исправления, исправленный diff или None)
        """
        changed = False
        final_diff = None
        
        try:
            diff = tc['function']['arguments']['diff']
            lines = diff.split('\n')
            separator_count = 0
            separator_pattern = re.compile(r'^[=<>-]{3,}$|^<<<<<<<|^>>>>>>>|^=======$|^-------$', re.IGNORECASE)
            
            for line in lines:
                if separator_pattern.match(line.strip()):
                    separator_count += 1
            
            if separator_count > 6:
                if debug:
                    self._log(f"⚠ Обнаружено {separator_count} разделителей, вызываем deep_search")
                
                file_content = data.get("content", {}) if data and data.get("type") == "file_content" else {}
                final_diff, issues = self._deep_search(diff, file_content, debug)
                
                if final_diff:
                    tc['function']['arguments']['diff'] = final_diff
                    changed = True
                    self._was_changed = True
                else:
                    # Ошибка в deep_search - сбрасываем лог
                    self._log("deep_search не смог исправить diff", "ERROR")
                    self._dump_log_on_error()
                
                return changed, final_diff
            
            if data and data.get("content"):
                if not all(isinstance(k, int) for k in data['content'].keys()):
                    self._log("Ключи content не int, пропускаем", "WARNING")
                    return changed, final_diff
            
            diff = tc['function']['arguments']['diff']
            lines = diff.split('\n')
            
            start_patterns = [
                r'^[\[\]<>=-]*\s*(?:SEARCH|SOURCE|SRC|REPLACE)\s*[\[\]<>=-]*$',
                r'^[\[\]<>=-]+$',
                r'^:(start|end)_line:',
            ]
            
            separator_indices = []
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                for pattern in start_patterns:
                    if re.match(pattern, line_stripped):
                        if debug and line_stripped != '=======':
                            self._log(f"  Заменяем разделитель на строке {i}: '{line_stripped}' -> '======='")
                        lines[i] = '======='
                        changed = True
                        separator_indices.append(i)
                        break
            
            i = 0
            while i < len(separator_indices) - 1:
                if separator_indices[i+1] == separator_indices[i] + 1:
                    if debug:
                        self._log(f"  Удаляем дублирующийся разделитель на строке {separator_indices[i]}")
                    del lines[separator_indices[i]]
                    for j in range(i+1, len(separator_indices)):
                        separator_indices[j] -= 1
                    del separator_indices[i]
                    changed = True
                else:
                    i += 1
            
            if not separator_indices or separator_indices[0] != 0:
                if debug:
                    self._log("  Добавляем разделитель в начало")
                lines.insert(0, '=======')
                separator_indices = [i+1 for i in separator_indices]
                separator_indices.insert(0, 0)
                changed = True
            
            if not separator_indices or separator_indices[-1] != len(lines) - 1:
                if debug:
                    self._log("  Добавляем разделитель в конец")
                lines.append('=======')
                separator_indices.append(len(lines) - 1)
                changed = True
            
            if len(separator_indices) == 3:
                if debug:
                    self._log("  Обнаружено 3 разделителя, добавляем :start_line: и разделитель")
                lines.insert(separator_indices[0] + 1, ':start_line:1')
                lines.insert(separator_indices[0] + 2, '-------')
                separator_indices = [
                    separator_indices[0],
                    separator_indices[0] + 2,
                    separator_indices[1] + 2,
                    separator_indices[2] + 2
                ]
                changed = True
            
            if len(separator_indices) != 4:
                if debug:
                    self._log(f"❌ Ошибка: найдено {len(separator_indices)} разделителей, нужно 4")
                self._log(f"Ошибка: найдено {len(separator_indices)} разделителей, нужно 4", "ERROR")
                self._dump_log_on_error()
                return changed, final_diff
            
            lines[separator_indices[0]] = '<<<<<<< SEARCH'
            
            lines[separator_indices[1]] = '-------'
            search_block = [separator_indices[1]+1, separator_indices[2]-separator_indices[1]-1]
            
            lines[separator_indices[2]] = '======='
            
            try:
                lines[separator_indices[3]] = '>>>>>>> REPLACE'
            except Exception:
                pass
            
            try:
                j = separator_indices[3] - 1
                while j > separator_indices[2] and not lines[j].strip():
                    del lines[j]
                    separator_indices[3] -= 1
                    j -= 1
                    changed = True
            except Exception:
                pass
            
            if data and data.get('content'):
                file_content = data['content']
                if search_block:
                    actual_line = self._find_search_block_line(file_content, lines, tuple(search_block))
                    if actual_line is not None:
                        for idx, line in enumerate(lines):
                            if line.strip().startswith(':start_line:'):
                                lines[idx] = f':start_line:{actual_line}'
                                changed = True
                                break
            
            tc['function']['arguments']['diff'] = '\n'.join(lines)
            final_diff = '\n'.join(lines)
            
            if changed:
                self._was_changed = True
            
            return changed, final_diff
            
        except Exception as e:
            self._log(f"Критическая ошибка в fix: {str(e)}", "ERROR")
            self._dump_log_on_error()
            raise
    
    def validate_diff_format(self, diff_lines: List[str]) -> Tuple[bool, List[str]]:
        """
        Валидирует формат diff.
        
        Ожидаемый формат:
        1. <<<<<<< SEARCH
        2. :start_line:N
        3. -------
        4. ... search content ...
        5. =======
        6. ... replace content ...
        7. >>>>>>> REPLACE
        """
        errors = []
        
        if not diff_lines:
            errors.append("Diff пуст")
            self._log("Diff пуст", "ERROR")
            self._dump_log_on_error()
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
        elif separator_idx == 3:
            errors.append("Разделитель '=======' не может быть на 4 строке (должен быть хотя бы одна строка SEARCH контента)")
        
        if diff_lines[-1] != '>>>>>>> REPLACE':
            errors.append(f"Последняя строка: ожидается '>>>>>>> REPLACE', получено '{diff_lines[-1]}'")
        
        if errors:
            for error in errors:
                self._log(f"Ошибка валидации: {error}", "ERROR")
            self._dump_log_on_error()
        
        return len(errors) == 0, errors
    
    def reset(self):
        """Сброс состояния"""
        self._was_changed = False
        self._log_buffer.clear()

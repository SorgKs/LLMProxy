# apply_diff_fixer.py
"""
Класс для исправления и валидации apply_diff tool calls.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

import handlers

logger = logging.getLogger(__name__)


class FixToolApplyDiff:
    """
    Класс для комплексного исправления apply_diff:
    форматирование и поиск start_line.
    """
    
    def __init__(self):
        self._was_changed = False
    
    @property
    def changed(self) -> bool:
        """Были ли изменения"""
        return self._was_changed
    
    def _print_diff(self, lines: List[str], title: str) -> None:
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
            print(f"\n{handlers.Colors.CYAN}{'='*60}{handlers.Colors.RESET}")
            print(f"{handlers.Colors.BOLD}Deep Search: Анализ проблемного diff{handlers.Colors.RESET}")
            print(f"{handlers.Colors.CYAN}{'='*60}{handlers.Colors.RESET}")
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
        clean_lines = remaining_lines
        clean_content = '\n'.join(clean_lines)
        
        if debug:
            print(f"\n{handlers.Colors.YELLOW}Чистый контент (без маркеров):{handlers.Colors.RESET}")
            print(f"{handlers.Colors.YELLOW}{'-'*40}{handlers.Colors.RESET}")
            for i, line in enumerate(clean_lines[:10]):
                repr_line = line.replace(' ', '·').replace('\t', '→')
                print(f"{i:3}: '{repr_line[:80]}'")
            if len(clean_lines) > 10:
                print(f"... и еще {len(clean_lines) - 10} строк")
            print(f"{handlers.Colors.YELLOW}{'-'*40}{handlers.Colors.RESET}")
        
        # Фаза 2: Поиск в файле
        if not file_content:
            issues.append("Нет содержимого файла для верификации")
            return None, issues
        
        # Берем первые 3 строки из чистого контента (как есть, включая пустые, сохраняя пробелы)
        search_prefix = clean_lines[:3]
        
        if not search_prefix:
            issues.append("Чистый контент пуст")
            return None, issues
        
        if debug:
            print(f"\n{handlers.Colors.CYAN}Поиск в файле первых 3 строк SEARCH блока:{handlers.Colors.RESET}")
            for i, line in enumerate(search_prefix):
                repr_line = line.replace(' ', '·').replace('\t', '→')
                print(f"  {i+1}: '{repr_line[:80]}'")
        
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
                    print(f"{handlers.Colors.GREEN}✓ Найдено ТОЧНОЕ совпадение на позиции {i} (строка {i+1} в файле){handlers.Colors.RESET}")
                break
            
            # Если точного нет, но trim совпал - запоминаем для предупреждения
            if trim_match and not match:
                whitespace_warning = True
                if debug:
                    print(f"{handlers.Colors.YELLOW}⚠ На позиции {i}: trim версии совпадают, но есть различия в пробелах{handlers.Colors.RESET}")
                    for j, search_line in enumerate(search_prefix[:2]):
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
        file_idx = found_position
        content_idx = 0
        true_separator = None
        true_separator_pos = None
        
        if debug:
            print(f"\n{handlers.Colors.CYAN}Поиск расхождения:{handlers.Colors.RESET}")
        
        while file_idx < len(file_lines_list) and content_idx < len(clean_lines):
            file_line = file_lines_list[file_idx]
            content_line = clean_lines[content_idx]
            
            if debug:
                file_repr = file_line.replace(' ', '·').replace('\t', '→')[:50]
                content_repr = content_line.replace(' ', '·').replace('\t', '→')[:50]
                print(f"  Сравнение: файл[{file_idx}]='{file_repr}' vs контент[{content_idx}]='{content_repr}'")
            
            if file_line != content_line:
                if debug:
                    print(f"{handlers.Colors.YELLOW}⚠ Расхождение на позиции {content_idx}{handlers.Colors.RESET}")
                
                if content_idx < len(clean_lines):
                    potential_separator = clean_lines[content_idx].strip()
                    for pattern in separator_patterns:
                        if re.match(pattern, potential_separator):
                            true_separator = potential_separator
                            true_separator_pos = content_idx
                            issues.append(f"Найден истинный разделитель: {true_separator} на позиции {true_separator_pos}")
                            if debug:
                                print(f"{handlers.Colors.GREEN}✓ Истинный разделитель: {true_separator}{handlers.Colors.RESET}")
                            break
                
                if true_separator is None:
                    issues.append(f"ОШИБКА: В месте расхождения нет разделителя (строка: '{clean_lines[content_idx][:50]}')")
                    return None, issues
                
                break
            
            file_idx += 1
            content_idx += 1
        
        if true_separator is None:
            issues.append("Не удалось найти истинный разделитель (достигнут конец файла или контента)")
            return None, issues
        
        # Фаза 4: Пересборка diff с экранированием ложных маркеров
        search_content_lines = clean_lines[:true_separator_pos]
        replace_content_lines = clean_lines[true_separator_pos + 1:] if true_separator_pos + 1 < len(clean_lines) else []
        
        if debug:
            print(f"\n{handlers.Colors.CYAN}Разделение контента:{handlers.Colors.RESET}")
            print(f"  SEARCH блок: {len(search_content_lines)} строк")
            print(f"  REPLACE блок: {len(replace_content_lines)} строк")
            print(f"  Истинный разделитель: '{true_separator}'")
        
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
            print(f"\n{handlers.Colors.GREEN}{handlers.Colors.BOLD}Финальный diff после deep_search:{handlers.Colors.RESET}")
            self._print_diff(final_lines, "Deep Search Result")
            print(f"\n{handlers.Colors.YELLOW}Проблемы и исправления:{handlers.Colors.RESET}")
            for issue in issues:
                print(f"  • {issue}")
        
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
        
        diff = tc['function']['arguments']['diff']
        lines = diff.split('\n')
        separator_count = 0
        separator_pattern = re.compile(r'^[=<>-]{3,}$|^<<<<<<<|^>>>>>>>|^=======$|^-------$', re.IGNORECASE)
        
        for line in lines:
            if separator_pattern.match(line.strip()):
                separator_count += 1
        
        if separator_count > 6:
            if debug:
                print(f"{handlers.Colors.YELLOW}⚠ Обнаружено {separator_count} разделителей, вызываем deep_search{handlers.Colors.RESET}")
            
            file_content = data.get("content", {}) if data and data.get("type") == "file_content" else {}
            final_diff, issues = self._deep_search(diff, file_content, debug)
            
            if final_diff:
                tc['function']['arguments']['diff'] = final_diff
                changed = True
                self._was_changed = True
            
            return changed, final_diff
        
        if data and data.get("content"):
            if not all(isinstance(k, int) for k in data['content'].keys()):
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
                        print(f"  Заменяем разделитель на строке {i}: '{line_stripped}' -> '======='")
                    lines[i] = '======='
                    changed = True
                    separator_indices.append(i)
                    break
        
        i = 0
        while i < len(separator_indices) - 1:
            if separator_indices[i+1] == separator_indices[i] + 1:
                if debug:
                    print(f"  Удаляем дублирующийся разделитель на строке {separator_indices[i]}")
                del lines[separator_indices[i]]
                for j in range(i+1, len(separator_indices)):
                    separator_indices[j] -= 1
                del separator_indices[i]
                changed = True
            else:
                i += 1
        
        if not separator_indices or separator_indices[0] != 0:
            if debug:
                print("  Добавляем разделитель в начало")
            lines.insert(0, '=======')
            separator_indices = [i+1 for i in separator_indices]
            separator_indices.insert(0, 0)
            changed = True
        
        if not separator_indices or separator_indices[-1] != len(lines) - 1:
            if debug:
                print("  Добавляем разделитель в конец")
            lines.append('=======')
            separator_indices.append(len(lines) - 1)
            changed = True
        
        if len(separator_indices) == 3:
            if debug:
                print("  Обнаружено 3 разделителя, добавляем :start_line: и разделитель")
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
                print(f"❌ Ошибка: найдено {len(separator_indices)} разделителей, нужно 4")
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
        
        return len(errors) == 0, errors
    
    def reset(self):
        """Сброс состояния"""
        self._was_changed = False
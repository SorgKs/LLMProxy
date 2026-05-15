# tests/test_apply_diff_fix.py
import json
import os
import sys
import re
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Теперь импортируем AnswerProcessor
from answers import AnswerProcessor

# Константы для цветов (упрощенно для pytest)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class TestApplyDiffFix:
    """Тест для проверки исправления apply_diff"""
    
    @classmethod
    def setup_class(cls):
        """Настройка для всего класса тестов"""
        cls.processor = AnswerProcessor()
        cls.test_file = os.path.join(os.path.dirname(__file__), 'assets', 'test_apply_diff_fix.json')
    
    def load_test_cases(self) -> List[Dict]:
        """Загружает тестовые случаи из JSON файла"""
        if not os.path.exists(self.test_file):
            pytest.skip(f"Файл с тестами не найден: {self.test_file}")
            return []
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                pytest.skip(f"Файл с тестами пуст: {self.test_file}")
                return []
            return json.loads(content)
    
    def normalize_diff(self, diff: str) -> str:
        """Нормализует diff для сравнения (убирает лишние пробелы в конце строк)"""
        lines = diff.split('\n')
        normalized = []
        for line in lines:
            normalized.append(line.rstrip())
        return '\n'.join(normalized)
    
    def validate_roo_format(self, diff: str) -> Dict[str, Any]:
        """Проверяет, соответствует ли diff формату RooCode."""
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "has_start_line": False,
            "has_search_marker": False,
            "has_replace_marker": False,
            "has_separator": False
        }
        
        # Проверка наличия обязательных маркеров
        if "<<<<<<< SEARCH" not in diff:
            result["errors"].append("Missing <<<<<<< SEARCH marker")
        else:
            result["has_search_marker"] = True
        
        if ">>>>>>> REPLACE" not in diff:
            result["errors"].append("Missing >>>>>>> REPLACE marker")
        else:
            result["has_replace_marker"] = True
        
        if "=======" not in diff:
            result["errors"].append("Missing ======= separator")
        else:
            result["has_separator"] = True
        
        # Проверка наличия start_line
        if ":start_line:" not in diff:
            result["errors"].append("Missing :start_line: directive")
        else:
            start_match = re.search(r':start_line:\s*(\d+)', diff)
            if start_match:
                result["has_start_line"] = True
                line_num = int(start_match.group(1))
                if line_num <= 0:
                    result["errors"].append(f"Invalid start_line: {line_num} (must be positive)")
            else:
                result["errors"].append("Invalid :start_line: format")
        
        # Проверка наличия разделителя -------
        if "-------" not in diff:
            result["warnings"].append("Missing ------- separator (optional but recommended)")
        
        # Проверка порядка секций
        if result["has_search_marker"] and result["has_separator"] and result["has_replace_marker"]:
            search_pos = diff.find("<<<<<<< SEARCH")
            sep_pos = diff.find("=======")
            replace_pos = diff.find(">>>>>>> REPLACE")
            
            if not (search_pos < sep_pos < replace_pos):
                result["errors"].append("Incorrect section order")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def test_all_cases(self):
        """Запускает все тестовые случаи"""
        test_cases = self.load_test_cases()
        if not test_cases:
            return
        
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЯ APPLY_DIFF{RESET}")
        print(f"{BOLD}{'='*80}{RESET}")
        print(f"\n📋 Загружено тестов: {len(test_cases)}")
        
        passed = 0
        failed = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{BOLD}{'='*60}{RESET}")
            print(f"{BOLD}ТЕСТ #{i}: {test_case.get('name', 'Без имени')}{RESET}")
            print(f"{BOLD}{'='*60}{RESET}")
            
            # === ТЕСТ БЕЗ DATA ===
            print(f"\n{BLUE}  --- Тест без data (сохранение оригинального start_line) ---{RESET}")
            try:
                args = {
                    "path": "test.py",
                    "diff": test_case["input"]["diff"]
                }
                
                was_changed = self.processor.fix_apply_diff(args, debug=False)
                
                actual_diff = args["diff"]
                expected_diff = test_case["expected"]["diff"]
                
                normalized_actual = self.normalize_diff(actual_diff)
                normalized_expected = self.normalize_diff(expected_diff)
                
                validation = self.validate_roo_format(actual_diff)
                
                if normalized_actual == normalized_expected and validation["valid"]:
                    print(f"{GREEN}  ✅ БЕЗ DATA: ПРОЙДЕН{RESET}")
                else:
                    print(f"{RED}  ❌ БЕЗ DATA: НЕ ПРОЙДЕН{RESET}")
                    failed += 1
                    
                    if not validation["valid"]:
                        print(f"{RED}     Ошибки формата:{RESET}")
                        for error in validation["errors"]:
                            print(f"{RED}       • {error}{RESET}")
                    
                    if normalized_actual != normalized_expected:
                        print(f"{YELLOW}     Результат не соответствует ожидаемому{RESET}")
                        
                        debug_args = {
                            "path": "test.py",
                            "diff": test_case["input"]["diff"]
                        }
                        self.processor.fix_apply_diff(debug_args, debug=True)
                        
                        actual_lines = normalized_actual.split('\n')
                        expected_lines = normalized_expected.split('\n')
                        
                        if len(actual_lines) != len(expected_lines):
                            print(f"{YELLOW}       Разное количество строк: actual={len(actual_lines)}, expected={len(expected_lines)}{RESET}")
                        
                        diff_count = 0
                        for line_idx, (a, e) in enumerate(zip(actual_lines, expected_lines)):
                            if a != e and diff_count < 3:
                                print(f"{YELLOW}       Строка {line_idx+1} отличается:{RESET}")
                                print(f"{YELLOW}         actual:   '{a}'{RESET}")
                                print(f"{YELLOW}         expected: '{e}'{RESET}")
                                diff_count += 1
                                if diff_count >= 3:
                                    print(f"{YELLOW}         ... (остальные различия скрыты){RESET}")
                                    break
            except Exception as e:
                print(f"{RED}  ❌ БЕЗ DATA: ОШИБКА: {str(e)}{RESET}")
                failed += 1
            
            # === ТЕСТ С DATA ===
            if "data" in test_case and "expected3" in test_case:
                print(f"\n{BLUE}  --- Тест с data (поиск актуального start_line) ---{RESET}")
                try:
                    args = {
                        "path": "test.py",
                        "diff": test_case["input"]["diff"]
                    }
                    
                    data = test_case["data"]
                    
                    # Сначала исправляем формат
                    self.processor.fix_apply_diff(args, debug=False)
                    
                    # Затем проверяем и исправляем start_line на основе данных файла
                    if data and data.get("type") == "file_content":
                        self.processor._validate_and_fix_apply_diff(args, data)
                    
                    actual_diff = args["diff"]
                    expected_diff = test_case["expected3"]["diff"]
                    
                    normalized_actual = self.normalize_diff(actual_diff)
                    normalized_expected = self.normalize_diff(expected_diff)
                    
                    validation = self.validate_roo_format(actual_diff)
                    
                    if normalized_actual == normalized_expected and validation["valid"]:
                        print(f"{GREEN}  ✅ С DATA: ПРОЙДЕН{RESET}")
                        passed += 1
                    else:
                        print(f"{RED}  ❌ С DATA: НЕ ПРОЙДЕН{RESET}")
                        failed += 1
                        
                        if not validation["valid"]:
                            print(f"{RED}     Ошибки формата:{RESET}")
                            for error in validation["errors"]:
                                print(f"{RED}       • {error}{RESET}")
                        
                        if normalized_actual != normalized_expected:
                            print(f"{YELLOW}     Результат не соответствует ожидаемому{RESET}")
                            
                            actual_lines = normalized_actual.split('\n')
                            expected_lines = normalized_expected.split('\n')
                            
                            if len(actual_lines) != len(expected_lines):
                                print(f"{YELLOW}       Разное количество строк: actual={len(actual_lines)}, expected={len(expected_lines)}{RESET}")
                            
                            diff_count = 0
                            for line_idx, (a, e) in enumerate(zip(actual_lines, expected_lines)):
                                if a != e and diff_count < 3:
                                    print(f"{YELLOW}       Строка {line_idx+1} отличается:{RESET}")
                                    print(f"{YELLOW}         actual:   '{a}'{RESET}")
                                    print(f"{YELLOW}         expected: '{e}'{RESET}")
                                    diff_count += 1
                                    if diff_count >= 3:
                                        print(f"{YELLOW}         ... (остальные различия скрыты){RESET}")
                                        break
                except Exception as e:
                    print(f"{RED}  ❌ С DATA: ОШИБКА: {str(e)}{RESET}")
                    failed += 1
            else:
                # Если нет data/expected3, считаем что тест без data прошел
                passed += 1
        
        # Выводим итоговую статистику
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}ИТОГИ ТЕСТИРОВАНИЯ{RESET}")
        print(f"{BOLD}{'='*80}{RESET}")
        print(f"Всего тестов: {len(test_cases) * 2}")
        print(f"{GREEN}✅ Пройдено: {passed}{RESET}")
        print(f"{RED}❌ Не пройдено: {failed}{RESET}")
        
        success_rate = round(passed / max(len(test_cases) * 2, 1) * 100, 2)
        if success_rate >= 80:
            color = GREEN
        elif success_rate >= 50:
            color = YELLOW
        else:
            color = RED
        print(f"{color}📊 Успешность: {success_rate}%{RESET}")
        
        # Проваливаем тест, если есть ошибки
        assert failed == 0, f"{failed} тестов не пройдено"


# Для запуска через pytest
def test_apply_diff_fix():
    """Основная функция тестирования для pytest"""
    tester = TestApplyDiffFix()
    tester.test_all_cases()


# Для запуска скрипта напрямую
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
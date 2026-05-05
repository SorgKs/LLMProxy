# tests/test_apply_diff_fix.py
import json
import os
import sys
import re
from typing import Optional, Dict, Any, List
from unittest.mock import MagicMock, patch

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Мокаем log.py ДО импорта answers.py
sys.modules['log'] = MagicMock()
import log
log.log_response = MagicMock()
log.log_tool_calls = MagicMock()
log.log_validation_error = MagicMock()

# Теперь импортируем AnswerProcessor
from answers import AnswerProcessor


class TestApplyDiffFix:
    """Тест для проверки исправления apply_diff"""
    
    # Default colors for terminal output
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
        "bold": "\033[1m"
    }

    def __init__(self):
        # Initialize instance attributes
        self.processor = AnswerProcessor()
        self.test_file = os.path.join(os.path.dirname(__file__), 'assets', 'test_apply_diff_fix.json')
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
    
    def setup_method(self, method):
        # Setup for each test method
        pass
    
    def c(self, color, text):
        """Возвращает цветной текст"""
        # Use class-level colors instead of instance colors
        return f"{self.__class__.colors.get(color, '')}{text}{self.__class__.colors['reset']}"
    
    def load_test_cases(self) -> List[Dict]:
        """Загружает тестовые случаи из JSON файла"""
        if not os.path.exists(self.test_file):
            print(self.c("red", f"❌ Файл с тестами не найден: {self.test_file}"))
            print(self.c("yellow", "   Создайте файл tests/assets/test_apply_diff_fix.json с тестовыми данными"))
            return []
        
        try:
            with open(self.test_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(self.c("red", f"❌ Файл с тестами пуст: {self.test_file}"))
                    return []
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(self.c("red", f"❌ Ошибка в JSON файле: {e}"))
            print(self.c("yellow", f"   Позиция ошибки: строка {e.lineno}, колонка {e.colno}"))
            if e.doc:
                lines = e.doc.splitlines()
                if e.lineno <= len(lines):
                    print(self.c("yellow", f"   Строка с ошибкой: {lines[e.lineno-1]}"))
            return []
    
    def normalize_diff(self, diff: str) -> str:
        """Нормализует diff для сравнения (убирает лишние пробелы в конце строк)"""
        lines = diff.split('\n')
        normalized = []
        for line in lines:
            normalized.append(line.rstrip())
        return '\n'.join(normalized)
    
    def create_tool_call(self, diff: str) -> Dict:
        """Создает tool call с фиксированным path для тестирования"""
        return {
            "function": {
                "name": "apply_diff",
                "arguments": {
                    "path": "test.py",  # фиксированный path, не влияет на тест
                    "diff": diff
                }
            }
        }
    
    def validate_roo_format(self, diff: str) -> Dict[str, Any]:
        """
        Проверяет, соответствует ли diff формату RooCode.
        Возвращает словарь с результатами проверки.
        """
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
    
    def run_test(self, test_case: Dict) -> Dict:
        """Запускает один тестовый случай"""
        name = test_case.get("name", "Unnamed test")
        input_data = test_case["input"]  # {"diff": "..."}
        expected = test_case["expected"]  # {"diff": "..."}
        
        result = {
            "name": name,
            "passed": False,
            "errors": [],
            "warnings": [],
            "input": input_data,
            "expected": expected,
            "actual": None
        }
        
        try:
            # Создаем tool call с тестируемым diff
            tool_call = self.create_tool_call(input_data["diff"])
            
            # Сохраняем исходный diff для отладки
            original_diff = tool_call['function']['arguments']['diff']
            
            # Сначала пробуем без debug
            was_changed = self.processor.fix_apply_diff(tool_call, debug=False)
            
            if not was_changed:
                # Проверяем, может быть diff уже был в правильном формате?
                validation = self.validate_roo_format(input_data["diff"])
                if validation["valid"]:
                    # Если diff уже валидный, то это тоже успех
                    result["actual"] = {"diff": input_data["diff"]}
                    result["passed"] = True
                    return result
            
            # Получаем исправленный diff
            actual_diff = tool_call['function']['arguments']['diff']
            result["actual"] = {"diff": actual_diff}
            
            # Проверяем валидность формата
            validation = self.validate_roo_format(actual_diff)
            if not validation["valid"]:
                result["errors"].extend(validation["errors"])
            
            # Нормализуем для сравнения
            normalized_actual = self.normalize_diff(actual_diff)
            normalized_expected = self.normalize_diff(expected["diff"])
            
            # Сравниваем с ожидаемым результатом
            if normalized_actual != normalized_expected:
                result["errors"].append("Результат не соответствует ожидаемому")
                
                # Если тест провален, запускаем с debug для отладки
                print(self.c("yellow", f"\n  🔍 Тест провален, повторный запуск с debug:"))
                
                # Создаем новый tool call
                debug_tool_call = self.create_tool_call(input_data["diff"])
                
                # Запускаем с debug
                self.processor.fix_apply_diff(debug_tool_call, debug=True)
                
                # Для отладки показываем различия
                actual_lines = normalized_actual.split('\n')
                expected_lines = normalized_expected.split('\n')
                
                if len(actual_lines) != len(expected_lines):
                    result["errors"].append(f"Разное количество строк: actual={len(actual_lines)}, expected={len(expected_lines)}")
                
                # Показываем первые несколько отличающихся строк
                diff_count = 0
                for i, (a, e) in enumerate(zip(actual_lines, expected_lines)):
                    if a != e and diff_count < 3:
                        result["errors"].append(f"Строка {i+1} отличается:")
                        result["errors"].append(f"  actual:   '{a}'")
                        result["errors"].append(f"  expected: '{e}'")
                        diff_count += 1
                        if diff_count >= 3:
                            result["errors"].append("  ... (остальные различия скрыты)")
                            break
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Исключение: {str(e)}")
        
        return result

    def run_all_tests(self):
        """Запускает все тесты"""
        print(self.c("bold", "=" * 80))
        print(self.c("bold", "ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЯ APPLY_DIFF"))
        print(self.c("bold", "=" * 80))
        
        test_cases = self.load_test_cases()
        if not test_cases:
            print(self.c("red", "\n❌ Нет тестов для запуска"))
            return
        
        self.results["total"] = len(test_cases)
        
        print(f"\n📋 Загружено тестов: {len(test_cases)}")
        
        for i, test_case in enumerate(test_cases, 1):
            print(self.c("bold", f"\n{'='*60}"))
            print(self.c("bold", f"ТЕСТ #{i}: {test_case.get('name', 'Без имени')}"))
            print(self.c("bold", f"{'='*60}"))
            
            result = self.run_test(test_case)
            self.results["details"].append(result)
            
            if result["passed"]:
                print(self.c("green", f"  ✅ ПРОЙДЕН"))
                self.results["passed"] += 1
            else:
                print(self.c("red", f"  ❌ НЕ ПРОЙДЕН"))
                self.results["failed"] += 1
                for error in result["errors"]:
                    print(self.c("red", f"     • {error}"))
        
        # Выводим итоговую статистику
        self.print_summary()
    
    def print_summary(self):
        """Выводит сводку по результатам тестов"""
        print(self.c("bold", "\n" + "=" * 80))
        print(self.c("bold", "ИТОГИ ТЕСТИРОВАНИЯ"))
        print(self.c("bold", "=" * 80))
        print(f"Всего тестов: {self.results['total']}")
        print(self.c("green" if self.results['passed'] > 0 else "reset", f"✅ Пройдено: {self.results['passed']}"))
        print(self.c("red" if self.results['failed'] > 0 else "reset", f"❌ Не пройдено: {self.results['failed']}"))
        
        success_rate = round(self.results["passed"] / max(self.results["total"], 1) * 100, 2)
        color = "green" if success_rate >= 80 else "yellow" if success_rate >= 50 else "red"
        print(self.c(color, f"📊 Успешность: {success_rate}%"))
        
        if self.results["failed"] > 0:
            print(self.c("yellow", "\n📝 Детали ошибок:"))
            for result in self.results["details"]:
                if not result["passed"]:
                    print(self.c("bold", f"\n  {result['name']}:"))
                    # Показываем только первые 3 ошибки для каждого теста
                    for i, error in enumerate(result["errors"]):
                        if i < 3:
                            print(self.c("red", f"    • {error}"))
                        else:
                            print(self.c("yellow", f"    • ... и еще {len(result['errors']) - 3} ошибок"))
                            break


def main():
    """Основная функция запуска тестов"""
    # Создаем временный объект для цветов в main
    class Temp:
        colors = {
            "green": "\033[92m",
            "red": "\033[91m", 
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }
        def c(self, color, text):
            return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    temp = Temp()
    print(temp.c("blue", "\n🔧 Запуск тестов исправления apply_diff...\n"))
    
    tester = TestApplyDiffFix()
    tester.run_all_tests()
    
    # Возвращаем код ошибки если есть упавшие тесты
    if tester.results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

def test_apply_diff_fix():
    tester = TestApplyDiffFix()
    tester.run_all_tests()
    assert tester.results['failed'] == 0
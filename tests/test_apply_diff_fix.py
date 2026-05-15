# tests/test_apply_diff_fix.py
import json
import os
import sys
import re
import logging
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Configure logging to show DEBUG messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

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
        cls.cases_dir = os.path.join(
            os.path.dirname(__file__), 'assets', 'apply_diff_fix'
        )

    def load_test_cases(self) -> List[Dict]:
        """Загружает тестовые случаи из директорий case_NNN"""
        if not os.path.isdir(self.cases_dir):
            import pytest
            pytest.skip(f"Директория с тестами не найдена: {self.cases_dir}")
            return []

        test_cases = []

        for case_name in sorted(os.listdir(self.cases_dir)):
            case_dir = os.path.join(self.cases_dir, case_name)
            if not os.path.isdir(case_dir):
                continue

            meta_file = os.path.join(case_dir, 'meta.json')
            request_file = os.path.join(case_dir, 'request.json')
            expected_file = os.path.join(case_dir, 'expected.json')
            data_file = os.path.join(case_dir, 'data.json')

            if not all(os.path.exists(f) for f in [meta_file, request_file, expected_file, data_file]):
                continue

            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            with open(request_file, 'r', encoding='utf-8') as f:
                request_data = json.load(f)

            with open(expected_file, 'r', encoding='utf-8') as f:
                expected = json.load(f)

            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Преобразуем строковые ключи в int для content
                if data.get("type") == "file_content" and "content" in data:
                    content_dict = data["content"]
                    if isinstance(content_dict, dict):
                        # Преобразуем ключи из строк в int
                        data["content"] = {int(k): v for k, v in content_dict.items()}

            test_cases.append({
                "name": case_name,
                "body": request_data.get("body", {}),
                "meta": meta,
                "expected": expected,
                "data": data
            })

        return test_cases

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

        if ">>>>>> REPLACE" not in diff:
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
            result["has_start_line"] = True
            start_match = re.search(r':start_line:\s*(\d+)', diff)
            if start_match:
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
            replace_pos = diff.find(">>>>>> REPLACE")

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

            # === ТЕСТ С DATA ===
            if test_case.get("data"):
                print(f"\n{BLUE}  --- Тест с data (поиск актуального start_line) ---{RESET}")
                args = {
                    "path": test_case["meta"].get("path", "test.py"),
                    "diff": test_case["body"].get("diff", "")
                }

                # Используем _fix_apply_diff_tool
                self.processor._fix_apply_diff_tool(args, test_case["data"], debug=False)

                actual_diff = args["diff"]
                print("========actual_diff===========")
                print(actual_diff)
                print("==============================")
                expected_diff = test_case["expected"].get("diff", "")

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

                        debug_args = {
                            "path": test_case["meta"].get("path", "test.py"),
                            "diff": test_case["body"].get("diff", "")
                        }
                        self.processor._fix_apply_diff_tool(debug_args, test_case["data"], debug=True)

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
            else:
                # Если нет data, пропускаем тест
                print(f"\n{YELLOW}  --- Нет data, тест пропущен ---{RESET}")
                continue

        # Выводим итоговую статистику
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}ИТОГИ ТЕСТИРОВАНИЯ{RESET}")
        print(f"{BOLD}{'='*80}{RESET}")
        print(f"Всего тестов: {len(test_cases)}")
        print(f"{GREEN}✅ Пройдено: {passed}{RESET}")
        print(f"{RED}❌ Не пройдено: {failed}{RESET}")

        success_rate = round(passed / max(len(test_cases), 1) * 100, 2)
        if success_rate >= 80:
            color = GREEN
        elif success_rate >= 50:
            color = YELLOW
        else:
            color = RED
        print(f"{color}📊 Успешность: {success_rate}%{RESET}")

        # Проваливаем тест, если есть ошибки
        assert failed == 0, f"{failed} тестов не пройдено"


# Для запуска скрипта напрямую
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])

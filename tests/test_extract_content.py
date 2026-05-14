# tests/test_extract_content.py
"""
Комбинированные тесты: extract + check_file_sufficiency/check_function_sufficiency на основе JSON-кейсов из assets.
"""
import log
import os
import sys
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from requests import _extract_file_content_from_request, check_file_sufficiency, check_function_sufficiency


class TestCombinedExtractAndCheck(unittest.TestCase):
    """Комбинированные тесты: сначала extract, потом check на существующих кейсах."""

    @classmethod
    def setUpClass(cls):
        """Загружаем тестовые случаи из директорий case_NNN."""
        cls.cases_dir = os.path.join(
            os.path.dirname(__file__), 'assets', 'extract_file_content_from_request'
        )
        cls.test_cases = []
        
        if not os.path.isdir(cls.cases_dir):
            return
        
        for case_name in sorted(os.listdir(cls.cases_dir)):
            case_dir = os.path.join(cls.cases_dir, case_name)
            if not os.path.isdir(case_dir):
                continue
            
            meta_file = os.path.join(case_dir, 'meta.json')
            expected_file = os.path.join(case_dir, 'expected.json')
            expected2_file = os.path.join(case_dir, 'expected2.json')
            request_file = os.path.join(case_dir, 'request.json')
            
            if not all(os.path.exists(f) for f in [meta_file, expected_file, expected2_file, request_file]):
                continue
            
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            with open(expected_file, 'r', encoding='utf-8') as f:
                expected = json.load(f)
            
            with open(expected2_file, 'r', encoding='utf-8') as f:
                expected2 = json.load(f)
            
            with open(request_file, 'r', encoding='utf-8') as f:
                request_data = json.load(f)
            
            # Конвертируем индексы content в int для expected
            if expected and "content" in expected:
                expected["content"] = {
                    int(k): v for k, v in expected["content"].items()
                }
            
            # Конвертируем индексы content в int для expected2
            if expected2 and "content" in expected2 and isinstance(expected2["content"], dict):
                expected2["content"] = {
                    int(k): v for k, v in expected2["content"].items()
                }
            
            cls.test_cases.append({
                "name": case_name,
                "body": request_data.get("body", {}),
                "target_path": meta.get("path"),
                "meta": meta,
                "expected": expected,
                "expected2": expected2,
            })

    def test_combined_extract_and_check(self):
        """Тестирует extract + check: для file_content и function_content."""
        if not self.test_cases:
            self.skipTest("Директория с тестовыми случаями пуста или не найдена")

        for case in self.test_cases:
            name = case.get("name", "Без имени")
            body = case["body"]
            target_path = case.get("target_path")
            meta = case.get("meta", {})
            expected = case.get("expected")
            expected2 = case.get("expected2")
            check_type = meta.get("type")
            
            with self.subTest(name=name):
                # Шаг 1: Extract
                extracted = _extract_file_content_from_request(body, target_path)
                
                # Проверка 1: Сравниваем extracted с expected.json
                if expected is None:
                    self.assertIsNone(extracted, f"Extract: ожидался None, получен {extracted}")
                else:
                    self.assertIsNotNone(extracted, f"Extract: ожидался не-None результат")
                    
                    # Сравниваем все поля, которые есть в expected
                    for key, expected_value in expected.items():
                        self.assertIn(key, extracted, f"Extract: поле '{key}' отсутствует в результате")
                        actual_value = extracted[key]
                        
                        if key == 'content' and isinstance(expected_value, dict):
                            # Для content сравниваем по ключам (номерам строк)
                            for line_num, expected_line in expected_value.items():
                                print(f"DEBUG: line_num={line_num}, type={type(line_num)}")
                                print(f"DEBUG: actual_value keys={list(actual_value.keys())}")
                                print(f"DEBUG: expected_value keys={list(expected_value.keys())}")
                                self.assertIn(line_num, actual_value, f"Extract: строка {line_num} отсутствует в content")
                                self.assertEqual(
                                    expected_line, actual_value[line_num],
                                    f"Extract: строка {line_num} не совпадает:\nОжидалось: {expected_line!r}\nПолучено:  {actual_value[line_num]!r}"
                                )
                        elif key == 'content' and isinstance(expected_value, list):
                            # Для content как списка
                            self.assertEqual(
                                len(actual_value), len(expected_value),
                                f"Extract: длина content не совпадает: {len(actual_value)} vs {len(expected_value)}"
                            )
                            for i, (exp_line, act_line) in enumerate(zip(expected_value, actual_value)):
                                self.assertEqual(
                                    exp_line, act_line,
                                    f"Extract: строка {i+1} не совпадает:\nОжидалось: {exp_line!r}\nПолучено:  {act_line!r}"
                                )
                        else:
                            # Для простых полей (type, path, EOF, line_count)
                            self.assertEqual(
                                expected_value, actual_value,
                                f"Extract: поле '{key}' не совпадает: ожидалось {expected_value}, получено {actual_value}"
                            )
                
                # Если extract вернул None и expected тоже None, то check делать не нужно
                if extracted is None and expected is None:
                    continue
                
                # Шаг 2: Check в зависимости от типа из meta.json
                if check_type == 'file_content':
                    result = check_file_sufficiency(extracted, meta)
                elif check_type == 'function_content':
                    function_name = meta.get('function')
                    if not function_name:
                        self.fail(f"В meta.json отсутствует поле 'function'")
                    result = check_function_sufficiency(extracted, function_name)
                else:
                    self.fail(f"Неизвестный тип: {check_type}")
                
                # Проверка 2: Сравниваем result с expected2.json
                print(f"DEBUG: expected2={expected2}")
                if expected2 is None:
                    self.assertIsNone(result, f"Check: ожидался None, получен {result}")
                else:
                    self.assertIsNotNone(result, f"Check: ожидался не-None результат")
                    
                    # Сравниваем только те поля, которые есть в expected2
                    print(f"DEBUG: result keys = {list(result.keys())}")
                    for key, expected_value in expected2.items():
                        print(f"DEBUG: processing key={key}, expected_value type={type(expected_value)}")

                        self.assertIn(key, result, f"Check: поле '{key}' отсутствует в результате")
                        actual_value = result[key]
                        print(f"DEBUG: actual_value type={type(actual_value)}, keys={list(actual_value.keys()) if isinstance(actual_value, dict) else 'not dict'}")
                        print(f"DEBUG: result[{key}]={result[key]}")
                        if key == 'content' and isinstance(expected_value, dict):
                            # expected_value уже с int ключами после конвертации в setUpClass
                            for line_num, expected_line in expected_value.items():
                                self.assertIn(line_num, actual_value, f"Check: строка {line_num} отсутствует в content")
                                self.assertEqual(
                                    expected_line, actual_value[line_num],
                                    f"Check: строка {line_num} не совпадает:\nОжидалось: {expected_line!r}\nПолучено:  {actual_value[line_num]!r}"
                                )
                        elif key == 'content' and isinstance(expected_value, list):
                            # Для content как списка строк
                            self.assertEqual(
                                len(actual_value), len(expected_value),
                                f"Check: длина content не совпадает: {len(actual_value)} vs {len(expected_value)}"
                            )
                            for i, (exp_line, act_line) in enumerate(zip(expected_value, actual_value)):
                                self.assertEqual(
                                    exp_line, act_line,
                                    f"Check: строка {i+1} не совпадает:\nОжидалось: {exp_line!r}\nПолучено:  {act_line!r}"
                                )
                        else:
                            # Для простых полей (type, path, line_count)
                            self.assertEqual(
                                expected_value, actual_value,
                                f"Check: поле '{key}' не совпадает: ожидалось {expected_value}, получено {actual_value}"
                            )

        if self.test_cases:
            print(f"\n✅ Проверено {len(self.test_cases)} комбинированных кейсов")


if __name__ == "__main__":
    unittest.main(verbosity=2)
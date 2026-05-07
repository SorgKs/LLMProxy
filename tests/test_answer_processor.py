# test_answer_processor.py
import json
import unittest
from unittest.mock import Mock
from answers import AnswerProcessor


class TestRealLLMResponse(unittest.TestCase):
    """Тест на реальном ответе LLM из resp_20260505_003328_00f03e3b_response.json"""
    
    def setUp(self):
        self.processor = AnswerProcessor()
        self.processor.workspace_path = "/home/sorg/ZFS/Dev/LLMProxy"
    
    def create_answer_from_real_response(self, response_data: dict):
        """Создает Answer из реального ответа LLM"""
        answer = Mock()
        answer.model = response_data.get("model", "unknown")
        answer.duration = response_data.get("duration_seconds", 1.0)
        answer.is_stream = False
        answer.status_code = response_data.get("status_code", 200)
        answer.full_response = response_data.get("response", {})
        
        # Извлекаем content и tool_calls
        choices = answer.full_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            answer.content = message.get("content", "")
            answer.tool_calls = message.get("tool_calls", [])
        else:
            answer.content = ""
            answer.tool_calls = []
        
        return answer
    
    def validate_tool_call(self, tc: dict) -> tuple:
        """Валидирует один tool call"""
        # 1. Проверяем структуру
        if 'function' not in tc:
            return False, "Missing 'function' field"
        
        func = tc['function']
        
        if 'name' not in func:
            return False, "Missing 'name' field"
        
        if 'arguments' not in func:
            return False, "Missing 'arguments' field"
        
        args = func['arguments']
        
        # 2. arguments должен быть строкой
        if not isinstance(args, str):
            return False, f"Arguments type is {type(args).__name__}, expected str"
        
        # 3. arguments должен быть валидным JSON
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        
        # 4. parsed должен быть dict (объектом), а не строкой
        if isinstance(parsed, str):
            return False, f"Parsed arguments is string, not object. Value: {parsed[:100]}"
        
        if not isinstance(parsed, dict):
            return False, f"Parsed arguments type is {type(parsed).__name__}, expected dict"
        
        # 5. Проверяем обязательные поля для read_file
        tool_name = func['name']
        if tool_name == "read_file":
            if "path" not in parsed:
                return False, f"Missing required field 'path'. Fields: {list(parsed.keys())}"
            if not isinstance(parsed["path"], str):
                return False, f"Field 'path' type is {type(parsed['path']).__name__}, expected str"
        
        return True, None
    
    def test_real_response_3e3b(self):
        """Тест на реальном ответе resp_20260505_003328_00f03e3b_response.json"""
        
        # ТОЧНЫЙ ответ LLM из файла 3e3b
        real_response = {
            "timestamp": "2026-05-05T00:33:28.462542",
            "status_code": 200,
            "duration_seconds": 9.149383783340454,
            "response": {
                "id": "gen-1777930399-9AlD6ykKsqaAEjr8jGal",
                "object": "chat.completion",
                "created": 1777930399,
                "model": "inclusionai/ling-2.6-1t-20260423:free",
                "provider": "Novita",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "index": 0,
                                    "id": "fc-b336f734-e044-46a9-a982-07cee265703a",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\": \"requests.py\"}"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        answer = self.create_answer_from_real_response(real_response)
        
        print("\n" + "="*60)
        print("📥 ДО PROCESS()")
        print("="*60)
        print(f"Model: {answer.model}")
        print(f"Tool calls count: {len(answer.tool_calls)}")
        
        for i, tc in enumerate(answer.tool_calls):
            print(f"\nTool call #{i+1}:")
            print(f"  Name: {tc['function']['name']}")
            print(f"  Arguments type: {type(tc['function']['arguments']).__name__}")
            print(f"  Arguments value: {tc['function']['arguments']}")
            
            # Валидация до process
            is_valid, error = self.validate_tool_call(tc)
            print(f"  Valid before process: {'✅' if is_valid else '❌'}")
            if error:
                print(f"  Error: {error}")
        
        # Запускаем процессор
        print("\n" + "="*60)
        print("🔄 ВЫЗОВ process()")
        print("="*60)
        self.processor.process(answer)
        
        print("\n" + "="*60)
        print("📤 ПОСЛЕ PROCESS()")
        print("="*60)
        
        # Проверяем изменения
        print(f"Были изменения: {self.processor.changed}")
        if self.processor.changes_log:
            print("Изменения:")
            for change in self.processor.changes_log:
                print(f"  - {change}")
        
        # Валидация после process
        print("\n" + "="*60)
        print("🔍 ВАЛИДАЦИЯ ПОСЛЕ PROCESS()")
        print("="*60)
        
        all_valid = True
        for i, tc in enumerate(answer.tool_calls):
            print(f"\nTool call #{i+1}:")
            print(f"  Name: {tc['function']['name']}")
            print(f"  Arguments type: {type(tc['function']['arguments']).__name__}")
            print(f"  Arguments value: {tc['function']['arguments']}")
            
            is_valid, error = self.validate_tool_call(tc)
            print(f"  Valid after process: {'✅' if is_valid else '❌'}")
            if error:
                print(f"  Error: {error}")
                all_valid = False
        
        # Проверяем итог
        print("\n" + "="*60)
        print(f"ИТОГ: {'✅ ВСЕ ВАЛИДНО' if all_valid else '❌ ЕСТЬ ОШИБКИ'}")
        print("="*60)
        
        self.assertTrue(all_valid, "Tool call невалиден после process()")
    
    def test_real_response_3e3b_arguments_structure(self):
        """Проверяет конкретную структуру arguments после process()"""
        
        real_response = {
            "response": {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "type": "function",
                            "index": 0,
                            "id": "fc-test",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"requests.py\"}"
                            }
                        }]
                    }
                }]
            }
        }
        
        answer = self.create_answer_from_real_response(real_response)
        self.processor.process(answer)
        
        # Получаем arguments после обработки
        args_str = answer.tool_calls[0]['function']['arguments']
        
        print(f"\nArguments после process(): {args_str}")
        
        # Парсим
        parsed = json.loads(args_str)
        
        # Проверяем, что это объект
        self.assertIsInstance(parsed, dict, "Arguments должен парситься в dict")
        
        # Проверяем наличие обязательных полей
        self.assertIn('path', parsed, "Отсутствует поле 'path'")
        self.assertIn('mode', parsed, "Отсутствует поле 'mode' (должно быть добавлено)")
        self.assertEqual(parsed['path'], 'requests.py', "Значение path изменилось")
        self.assertEqual(parsed['mode'], 'slice', "mode должен быть 'slice'")
        
        print(f"✅ parsed = {parsed}")
    
    def test_processor_adds_mode_slice(self):
        """Проверяет, что процессор добавляет mode='slice' к read_file"""
        
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"test.py\"}"
            }
        }]
        
        answer = self.create_answer_from_real_response({"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}})
        
        # До process
        args_before = answer.tool_calls[0]['function']['arguments']
        parsed_before = json.loads(args_before)
        self.assertNotIn('mode', parsed_before, "mode не должен быть до process")
        
        # Запускаем процессор
        self.processor.process(answer)
        
        # После process
        args_after = answer.tool_calls[0]['function']['arguments']
        parsed_after = json.loads(args_after)
        
        self.assertIn('mode', parsed_after, "mode должен появиться после process")
        self.assertEqual(parsed_after['mode'], 'slice', "mode должен быть 'slice'")
        self.assertEqual(parsed_after['path'], 'test.py', "path не должен измениться")
        
        print(f"\n✅ До: {parsed_before}")
        print(f"✅ После: {parsed_after}")
    
    def test_processor_fixes_double_encoding(self):
        """Проверяет, что процессор исправляет двойное экранирование"""
        
        # Симулируем проблему из f8e11a25 (двойное экранирование)
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "read_file",
                "arguments": "\"{\\\"path\\\": \\\"requests.py\\\"}\""
            }
        }]
        
        answer = self.create_answer_from_real_response({"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}})
        
        print(f"\n📥 До исправления: {answer.tool_calls[0]['function']['arguments']}")
        
        # Запускаем процессор
        self.processor.process(answer)
        
        print(f"📤 После исправления: {answer.tool_calls[0]['function']['arguments']}")
        
        # Проверяем, что теперь это валидный JSON
        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)
        
        self.assertIsInstance(parsed, dict, "Должен парситься в dict")
        self.assertEqual(parsed.get('path'), 'requests.py', "path должен сохраниться")
        self.assertEqual(parsed.get('mode'), 'slice', "mode должен быть добавлен")
        
        print(f"✅ Распарсено: {parsed}")

    def test_processor_search_files_double_encoding(self):
        """Проверяет, что процессор исправляет двойное экранирование для search_files"""
        
        # Симулируем проблему двойного экранирования для search_files
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "search_files",
                "arguments": "\"{\\\"path\\\": \\\".\\\", \\\"regex\\\": \\\\\\\"RequestProcessor\\\\\\\"}\""
            }
        }]
        
        answer = self.create_answer_from_real_response({"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}})
        
        print(f"\n📥 До исправления: {answer.tool_calls[0]['function']['arguments']}")
        
        # Запускаем процессор
        self.processor.process(answer)
        
        print(f"📤 После исправления: {answer.tool_calls[0]['function']['arguments']}")
        
        # Проверяем, что теперь это валидный JSON
        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)
        
        self.assertIsInstance(parsed, dict, "Должен парситься в dict")
        self.assertEqual(parsed.get('path'), '.', "path должен быть '.'")
        self.assertEqual(parsed.get('regex'), 'RequestProcessor', "regex должен быть сохранен")
        
        print(f"✅ Распарсено: {parsed}")

    def test_processor_list_files_double_encoding(self):
        """Проверяет, что процессор исправляет двойное экранирование для list_files"""
        
        # Симулируем проблему двойного экранирования для list_files
        tool_calls = [{
            "type": "function",
            "index": 0,
            "id": "fc-test",
            "function": {
                "name": "list_files",
                "arguments": "\"{\\\"path\\\": \".\", \\\"recursive\\\": true}\""
            }
        }]
        
        answer = self.create_answer_from_real_response({"response": {"choices": [{"message": {"tool_calls": tool_calls}}]}})
        
        print(f"\n📥 До исправления: {answer.tool_calls[0]['function']['arguments']}")
        
        # Запускаем процессор
        self.processor.process(answer)
        
        print(f"📤 После исправления: {answer.tool_calls[0]['function']['arguments']}")
        
        # Проверяем, что теперь это валидный JSON
        args_str = answer.tool_calls[0]['function']['arguments']
        parsed = json.loads(args_str)
        
        self.assertIsInstance(parsed, dict, "Должен парситься в dict")
        self.assertEqual(parsed.get('path'), '.', "path должен быть '.'")
        self.assertEqual(parsed.get('recursive'), True, "recursive должен быть True")
        
        print(f"✅ Распарсено: {parsed}")


def run_tests():
    """Запускает все тесты с выводом результата"""
    print("\n" + "🧪" * 30)
    print("ТЕСТИРОВАНИЕ ОБРАБОТЧИКА ОТВЕТОВ")
    print("🧪" * 30)
    
    # Создаем тестовый набор
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRealLLMResponse)
    
    # Запускаем с verbosity=2 для детального вывода
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
    else:
        print(f"❌ ПРОВАЛЕНО: {len(result.failures)} failures, {len(result.errors)} errors")
        print(f"   Всего тестов: {result.testsRun}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
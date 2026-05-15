import json
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from answers import AnswerProcessor

def run_case_011_debug():
    """Запускает _fix_apply_diff_tool для case_011 с debug выводом"""
    
    processor = AnswerProcessor()
    
    # Загружаем данные теста
    case_dir = 'tests/assets/apply_diff_fix/case_011'
    
    with open(os.path.join(case_dir, 'meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    with open(os.path.join(case_dir, 'request.json'), 'r', encoding='utf-8') as f:
        request_data = json.load(f)
    
    with open(os.path.join(case_dir, 'expected.json'), 'r', encoding='utf-8') as f:
        expected = json.load(f)
    
    with open(os.path.join(case_dir, 'data.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Преобразуем строковые ключи в int для content
    if data.get("type") == "file_content" and "content" in data:
        content_dict = data["content"]
        if isinstance(content_dict, dict):
            data["content"] = {int(k): v for k, v in content_dict.items()}
    
    # Формируем аргументы (arguments - это JSON строка, парсим её)
    arguments_str = request_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    arguments = json.loads(arguments_str)
    
    args = {
        "path": meta.get("path", "test.py"),
        "diff": arguments["diff"]
    }
    
    print("=" * 80)
    print("ЗАГОЛОВОК ИСХОДНОГО DIFF:")
    print("=" * 80)
    print(args["diff"])
    print("=" * 80)
    print()
    
    # Вызываем _fix_apply_diff_tool с debug=True
    print("=" * 80)
    print("ЗАПУСК _fix_apply_diff_tool С DEBUG=True:")
    print("=" * 80)
    processor._fix_apply_diff_tool(args, data, debug=True)
    
    print()
    print("=" * 80)
    print("ФИНАЛЬНЫЙ DIFF:")
    print("=" * 80)
    print(args["diff"])
    print("=" * 80)
    
    print()
    print("=" * 80)
    print("ОЖИДАЕМЫЙ DIFF:")
    print("=" * 80)
    print(expected.get("diff", ""))
    print("=" * 80)

if __name__ == "__main__":
    run_case_011_debug()
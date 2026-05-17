#!/usr/bin/env python3
"""
Извлекает и форматирует diff из JSON-файлов с tool_calls apply_diff
"""

import json
import sys
import argparse
from pathlib import Path


def extract_diff_from_file(filepath):
    """
    Извлекает diff из JSON-файла, содержащего tool_calls с apply_diff
    
    Args:
        filepath: путь к JSON-файлу
        
    Returns:
        str: извлечённый diff или None, если не найден
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Проходим по choices
    for choice in data.get('choices', []):
        message = choice.get('message', {})
        tool_calls = message.get('tool_calls', [])
        
        for tool_call in tool_calls:
            function = tool_call.get('function', {})
            if function.get('name') == 'apply_diff':
                arguments = function.get('arguments', '')
                if arguments:
                    try:
                        args_dict = json.loads(arguments)
                        diff = args_dict.get('diff', '')
                        if diff:
                            return diff
                    except json.JSONDecodeError:
                        # Если arguments не валидный JSON, пробуем извлечь diff через regex
                        import re
                        match = re.search(r'"diff":\s*"((?:[^"\\]|\\.)*)"', arguments)
                        if match:
                            # Раскодируем escape-последовательности
                            diff = match.group(1).encode().decode('unicode-escape')
                            return diff
    return None


def format_diff_for_display(diff):
    """
    Форматирует diff для красивого отображения на экране
    
    Args:
        diff: строка с diff-содержимым
        
    Returns:
        str: отформатированный diff
    """
    # Заменяем \n на реальные переносы строк
    formatted = diff.replace('\\n', '\n')
    
    # Удаляем лишние экранирования кавычек
    formatted = formatted.replace('\\"', '"')
    
    # Удаляем экранирование обратного слэша (если есть)
    formatted = formatted.replace('\\\\', '\\')
    
    return formatted


def print_diff(diff, filename=None, show_headers=True):
    """
    Выводит diff на экран в отформатированном виде
    
    Args:
        diff: строка с diff-содержимым
        filename: имя файла (для заголовка)
        show_headers: показывать ли заголовок
    """
    if show_headers:
        print("=" * 80)
        if filename:
            print(f"📄 Файл: {filename}")
        print("=" * 80)
        print()
    
    formatted_diff = format_diff_for_display(diff)
    print(formatted_diff)
    
    if show_headers:
        print()
        print("=" * 80)
        print(f"✅ Diff извлечён успешно (длина: {len(diff)} символов)")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Извлекает и отображает diff из JSON-файлов с tool_calls apply_diff'
    )
    parser.add_argument(
        'files', 
        nargs='+', 
        help='Пути к JSON-файлам (можно указать несколько)'
    )
    parser.add_argument(
        '-r', '--raw', 
        action='store_true',
        help='Показать "сырой" diff без замены \\n и прочих escape-последовательностей'
    )
    parser.add_argument(
        '-o', '--output', 
        help='Сохранить результат в файл (иначе выводится на экран)'
    )
    
    args = parser.parse_args()
    
    all_diffs = []
    
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"❌ Файл не найден: {filepath}", file=sys.stderr)
            continue
        
        try:
            diff = extract_diff_from_file(filepath)
            if diff:
                all_diffs.append((filepath, diff))
            else:
                print(f"⚠️  Diff не найден в файле: {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Ошибка при обработке {filepath}: {e}", file=sys.stderr)
    
    if not all_diffs:
        print("❌ Не найдено ни одного diff в указанных файлах", file=sys.stderr)
        sys.exit(1)
    
    # Сохраняем или выводим результат
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, (filename, diff) in enumerate(all_diffs):
                if i > 0:
                    f.write("\n" + "=" * 80 + "\n\n")
                if args.raw:
                    f.write(diff)
                else:
                    f.write(format_diff_for_display(diff))
        print(f"✅ Результат сохранён в: {args.output}")
    else:
        for filename, diff in all_diffs:
            if args.raw:
                print_diff(diff, filename, show_headers=(len(all_diffs) > 1))
            else:
                # Для отображения на экране всегда форматируем
                print_diff(diff, filename, show_headers=(len(all_diffs) > 1))


if __name__ == "__main__":
    main()
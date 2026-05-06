import re
from typing import List, Optional, Tuple

def debug_apply_diff_step_by_step(diff_text: str):
    """Пошаговый вывод алгоритма _apply_diff_algorithm"""
    lines = diff_text.split('\n')
    separator_patterns = [
        r'^={3,}$', r'^-{3,}$', r'^>{3,}$', r'^<{3,}$',
        r'^[\[{]-{3,}[\]}]$', r'^[\[{]={3,}[\]}]$', r'^[\[{]>{3,}[\]}]$', r'^[\[{]<{3,}[\]}]$',
        r'^[\[\[{]+SEARCH[\]\]]+$', r'^[\[\[{]+SOURCE[\]\]]+$', r'^[\[\[{]+SRC[\]\]]+$',
        r'^SEARCH$', r'^SOURCE$', r'^SRC$',
        r'^<<<<<<< SEARCH$', r'^<<<<<<< SOURCE$', r'^<<<<<<< SRC$',
        r'^[\[\[{]+REPLACE[\]\]]+$', r'^REPLACE$', r'^=======$',
        r'^[\[\[{]+=======[\]\]]+$',
        r'^>+ REPLACE$', r'^>+REPLACE$',
        r'^>>>>>>> REPLACE$', r'^>>>>>> REPLACE$', r'^>>>>>>>>>>>> REPLACE$'
    ]
    
    separator_indices = []
    changed = False
    
    def print_state(step_num, description):
        print(f'\n{"="*60}')
        print(f'ШАГ {step_num}: {description}')
        print(f'{"="*60}')
        print(f'Индексы разделителей: {separator_indices}')
        for i, line in enumerate(lines):
            marker = ''
            if i in separator_indices:
                marker = '  <-- sep'
            print(f'  {i:2}: {repr(line)}{marker}')
    
    # Шаг 1
    print(f'\n{"="*60}')
    print('ШАГ 1: Замена всех разделителей на =======')
    print(f'{"="*60}')
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        for pattern in separator_patterns:
            if re.match(pattern, line_stripped):
                old = lines[i]
                lines[i] = '======='
                if old != '=======':
                    print(f'  Строка {i}: {repr(old)} -> =======')
                separator_indices.append(i)
                changed = True
                break
    print_state(1, 'После замены всех разделителей')
    
    # Шаг 2
    i = 0
    while i < len(separator_indices) - 1:
        if separator_indices[i+1] == separator_indices[i] + 1:
            del lines[separator_indices[i]]
            for j in range(i+1, len(separator_indices)):
                separator_indices[j] -= 1
            del separator_indices[i]
            changed = True
        else:
            i += 1
    print_state(2, 'После удаления дубликатов')
    
    # Шаг 3
    if not separator_indices or separator_indices[0] != 0:
        lines.insert(0, '=======')
        separator_indices = [i+1 for i in separator_indices]
        separator_indices.insert(0, 0)
        changed = True
    print_state(3, 'После добавления в начало')
    
    # Шаг 4
    if not separator_indices or separator_indices[-1] != len(lines) - 1:
        lines.append('=======')
        separator_indices.append(len(lines) - 1)
        changed = True
    print_state(4, 'После добавления в конец')
    
    # Шаг 5
    if len(separator_indices) == 3:
        lines.insert(separator_indices[0] + 1, ':start_line:1')
        lines.insert(separator_indices[0] + 2, '-------')
        separator_indices = [
            separator_indices[0],
            separator_indices[0] + 2,
            separator_indices[1] + 2,
            separator_indices[2] + 2
        ]
        changed = True
    print_state(5, 'После добавления :start_line: и -------')
    
    # Шаг 6
    if len(separator_indices) != 4:
        print(f'\n❌ ОШИБКА: {len(separator_indices)} разделителей, нужно 4')
        return None
    print_state(6, 'Проверка: должно быть 4 разделителя')
    
    # Шаг 7
    old = lines[separator_indices[0]]
    lines[separator_indices[0]] = '<<<<<<< SEARCH'
    print(f'  sep[{separator_indices[0]}] {repr(old)} -> <<<<<<< SEARCH')
    
    old = lines[separator_indices[1]]
    lines[separator_indices[1]] = '-------'
    print(f'  sep[{separator_indices[1]}] {repr(old)} -> -------')
    
    old = lines[separator_indices[2]]
    lines[separator_indices[2]] = '======='
    print(f'  sep[{separator_indices[2]}] {repr(old)} -> =======')
    
    old = lines[separator_indices[3]]
    lines[separator_indices[3]] = '>>>>>>> REPLACE'
    print(f'  sep[{separator_indices[3]}] {repr(old)} -> >>>>>>> REPLACE')
    
    print_state(7, 'После замены на правильные маркеры')
    
    print(f'\n{"="*60}')
    print('ИТОГОВЫЙ РЕЗУЛЬТАТ:')
    print(f'{"="*60}')
    for i, line in enumerate(lines):
        print(f'  {i:2}: {repr(line)}')
    
    return '\n'.join(lines)

# Тестовые случаи с ожидаемым результатом
test_cases = [
    {
        "name": "Тест #1: [SEARCH] + [REPLACE]",
        "input": """=======
[SEARCH]
:start_line:462
-------
            # Оптимизируем описания инструментов:
            # - Удаляет все описания параметров
            # - Удаляет indentation полностью
            # - Удаляет mode полностью
            # - Оставляет только path в required
            # - Упрощает description функции
[REPLACE]
            # Оптимизируем описания инструментов:
            # - Удаляет все описания параметров
            # - Удаляет indentation полностью
            # - Удаляет mode полностью для всех режимов кроме orchestartor
            # - Оставляет только path в required
            # - Упрощает description функции
>>>>>>> REPLACE"""
    },
    {
        "name": "Тест #2: SEARCH + REPLACE без скобок",
        "input": """=======
SEARCH
:start_line:464
-------
            # Оптимизируем описания инструментов:
            # - Удаляет все описания параметров
            # - Удаляет indentation полностью
            # - Удаляет mode полностью
            # - Оставляет только path в required
            # - Упрощает description функции
            =======
REPLACE
            # Оптимизируем описания инструментов:
            # - Удаляет все описания параметров
            # - Удаляет indentation полностью
            # - Удаляет mode полностью для всех режимов кроме orchestartor
            # - Оставляет только path в required
            # - Упрощает description функции
            >>>>>>> REPLACE"""
    },
    {
        "name": "Тест #4: БЕЗ маркеров",
        "input": """:start_line:462
-------
            # Оптимизируем описания инструментов:
            # - Удаляет все описания параметров
            # - Удаляет indentation полностью
            # - Удаляет mode полностью
            # - Оставляет только path в required
            # - Упрощает description функции
-------
            # Оптимизируем описания инструментов:
            # - Удаляет все описания параметров
            # - Удаляет indentation полностью
            # - Удаляет mode полностью для всех режимов кроме orchestartor
            # - Оставляет только path в required
            # - Упрощает description функции"""
    }
]

for tc in test_cases:
    print(f'\n{"#"*80}')
    print(f'# {tc["name"]}')
    print(f'{"#"*80}')
    result = debug_apply_diff_step_by_step(tc['input'])
    if result:
        print(f'\n✓ Сгенерировано успешно')
    else:
        print(f'\n❌ Сбой генерации')

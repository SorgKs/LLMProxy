# Комплексный тест AnswerProcessor.process()

## Обзор

Тест `test_process_integration.py` проверяет полную интеграцию метода `AnswerProcessor.process()` без моков - все функции из модуля `answers.py` и `requests.py` вызываются напрямую.

## Типы тестов

### Двухэтапный тест (когда нужны данные файла)
1. **Этап 1**: `process(answer, None)` → ожидается `action: request_file` или `action: request_function`
2. **Этап 2**: Формируем `data` через реальные функции `requests.py`
3. **Этап 3**: `process(answer, data)` → проверяем исправленный результат

### Одноэтапный тест (когда сразу готов ответ)
- **Этап 1**: `process(answer, None)` → сразу получаем готовый ответ
- **Проверка**: модифицированный `answer.tool_calls`

## Структура тестовых данных

### Директория: tests/assets/process_integration/

#### Формат файлов case_NNN/

**request.json** - ПОЛНЫЙ запрос от клиента:

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Строку old замени на new в файле test.py"}
  ]
}
```

**answer.json** - ПОЛНЫЙ готовый объект ответа LLM:

```json
{
  "model": "gpt-4",
  "duration": 1.5,
  "is_stream": false,
  "status_code": 200,
  "content": "",
  "tool_calls": [{
    "type": "function",
    "id": "fc-001",
    "index": 0,
    "function": {
      "name": "apply_diff",
      "arguments": {
        "path": "test.py",
        "diff": "=======\nold\n=======\nnew\n>>>>>> REPLACE"
      }
    }
  }],
  "full_response": {
    "choices": [{
      "message": {
        "tool_calls": [{
          "type": "function",
          "id": "fc-001",
          "index": 0,
          "function": {
            "name": "apply_diff",
            "arguments": "{\"path\": \"test.py\", \"diff\": \"...\"}"
          }
        }]
      }
    }]
  }
}
```

**expected.json** - ПОЛНЫЙ исправленный ответ LLM after process():

Для apply_diff (двухэтапный):
```json
{
  "diff": "<<<<<<< SEARCH\n:start_line:1\n-------\nold\n=======\nnew\n>>>>>>> REPLACE"
}
```

Для request_file (двухэтапный, первый этап):
```json
{
  "action": "request_file",
  "path": "test.py"
}
```

Для read_file (одноэтапный):
```json
{
  "check": {
    "mode": "slice"
  }
}
```

## Логика работы теста

### Шаг 1: Загрузка тестовых случаев
- Сканирует директорию `tests/assets/process_integration/`
- Находит все поддиректории `case_NNN`
- Для каждого case читает файлы: `request.json`, `answer.json`, `expected.json`

### Шаг 2: Первый вызов process() без data
- Берет **ГОТОВЫЙ object `answer`** из `answer.json`
- Вызывает `processor.process(answer, None)`

### Шаг 3: Анализ результата

**Для ДВУХЭТАПНОГО теста:**
Если `result` содержит `action: request_file` или `action: request_function`:
- Запоминает `path` и `function_name` (если есть)
- Переходит к формированию data

**Для ОДНОЭТАПНОГО теста:**
Если `result is None`:
- Проверяет `answer.tool_calls[0].function.arguments`
- Сравнивает с `expected.diff` или `expected.check`
- **ТЕСТ ПРОЙДЕН** если совпадает

### Шаг 4: Формирование data (только для двухэтапного)
- На основе `result['action']`:
  - `request_file` → вызывает реальную функцию `request_file_content(path)`
  - `request_function` → вызывает реальную функцию `request_function_content(path, function_name)`
- **НЕ моки** - реальные функции из `requests.py`

### Шаг 5: Второй вызов process() со сформированным data
- Вызывает `processor.process(answer, data)`
- `data` получено на Шаге 4

### Шаг 6: Проверка результата второго вызова (для двухэтапного)
- Если `result is None`:
  - Извлекает `diff` из `answer.tool_calls[0].function.arguments`
  - Сравнивает с `expected.diff`
  - Проверяет формат RooCode
  - **ТЕСТ ПРОЙДЕН** если совпадает

## Примеры тестовых случаев

### Case 001: Двухэтапный тест (apply_diff)

request.json:
```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Строку old замени на new"}]
}
```

answer.json:
```json
{
  "tool_calls": [{
    "type": "function",
    "id": "fc-001",
    "function": {
      "name": "apply_diff",
      "arguments": {
        "path": "test.py",
        "diff": "=======\nold\n=======\nnew\n>>>>>> REPLACE"
      }
    }
  }]
}
```

expected.json:
```json
{
  "diff": "<<<<<<< SEARCH\n:start_line:1\n-------\nold\n=======\nnew\n>>>>>>> REPLACE"
}
```

**Алгоритм:**
1. process(answer, None) → {"action": "request_file", "path": "test.py"}
2. request_file_content("test.py") → data с содержимым файла
3. process(answer, data) → None, diff исправлен

### Case 002: Одноэтапный тест (read_file)

request.json:
```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Прочитай файл test.py"}]
}
```

answer.json:
```json
{
  "tool_calls": [{
    "type": "function",
    "id": "fc-002",
    "function": {
      "name": "read_file",
      "arguments": {"path": "test.py"}
    }
  }]
}
```

expected.json:
```json
{
  "check": {
    "mode": "slice"
  }
}
```

**Алгоритм:**
1. process(answer, None) → None
2. Проверяем answer.tool_calls[0].function.arguments['mode'] == 'slice'

## Методы тестового класса

### load_test_cases()
Сканирует директорию process_integration/ и загружает все case_NNN.

### get_data_from_request(action, path, function_name)
Формирует data через реальные функции из `requests.py` (без моков).

### normalize_diff(diff)
Убирает trailing spaces для корректного сравнения.

### validate_roo_format(diff)
Проверяет наличие маркеров:
- <<<<<<< SEARCH
- =======
- >>>>>>> REPLACE
- :start_line:

## Источник данных и параметры

### Где берутся данные для тестов

**request.json (ПОЛНЫЙ запрос):**
- Берется из реального HTTP запроса, приходящего от клиента в proxy.py
- Содержит: model, messages (с историей диалога), temperature, max_tokens и другие параметры

**answer.json (ПОЛНЫЙ ответ LLM):**
- Берется из реального ответа LLM, который приходит в proxy.py
- Содержит: model, duration, status_code, tool_calls, full_response
- Объект `answer` - это готовый объект с атрибутами (не словарь!)

**expected.json (ПОЛНЫЙ исправленный ответ LLM):**
- Содержит ожидаемый результат после обработки process()
- Для apply_diff: исправленный diff в формате RooCode
- Для read_file: проверка изменений в аргументах (mode, slice, и т.д.)

### Параметры для вызова process()

**process(answer, data):**
- `answer` - объект с атрибутами: tool_calls, full_response, model, duration, status_code
- `data` - None (для первого вызова) или dict с данными файла

### Как формируется data через requests.py

Для двухэтапного теста:
1. Первый вызов `process(answer, None)` возвращает `{"action": "request_file", "path": "..."}`
2. Вызываем реальную функцию `get_data_from_request(action, path, function_name)` из requests.py
3. Получаем: `{"type": "file_content", "path": "...", "content": {номер_строки: текст}}`
4. Второй вызов `process(answer, data)` с этим data

#### Функции из requests.py для формирования data

**`check_file_sufficiency(data: dict, pending_info: dict) -> Optional[dict]`**

Проверяет, содержит ли data полное содержимое целевого файла.

**Параметры:**
- `data` - Словарь с данными (тип file_content), содержащий:
  - `type` - "file_content"
  - `path` - путь к файлу
  - `content` - {номер_строки: текст}
  - `EOF` - флаг завершения файла
  - `line_count` - количество строк
- `pending_info` - Информация о запрошенном файле:
  - `path` - путь к файлу
  - `function_name` - название функции (если нужно)
  - `full_code` - флаг полного кода

**Возвращает:** Готовый словарь с данными файла или None

**`check_function_sufficiency(data: dict, function_name: str) -> Optional[str]`**

Проверяет, содержит ли data полное определение требуемой функции.

**Параметры:**
- `data` - Словарь с данными файла (см. выше)
- `function_name` - название функции для поиска

**Возвращает:** Словарь с содержимым функции или None

### Как проверяется результат

**Для двухэтапного теста:**
```python
# После второго вызова process() с data
assert result is None  # process() должен вернуть None
actual_diff = answer.tool_calls[0].function.arguments["diff"]
assert normalize_diff(actual_diff) == normalize_diff(expected["diff"])
assert validate_roo_format(actual_diff)["valid"] == True
```

**Для одноэтапного теста:**
```python
# После первого вызова process()
assert result is None  # process() должен вернуть None
args = answer.tool_calls[0].function.arguments
if expected.get("check"):
    for key, value in expected["check"].items():
        assert args.get(key) == value
# Механизм запроса и обработки данных между proxy.py и answer_processor

## Общая концепция

Механизм позволяет `answer_processor` (answers.py) запрашивать дополнительные данные у `proxy.py` (например, содержимое файла для преобразования `function_replace` → `apply_diff`), когда для выполнения операции не хватает контекста.

## Текущая реализация

### 1. Process возвращает действие

Когда `answer_processor.process()` не может выполнить преобразование из-за отсутствия данных:

```python
# В answers.py, process():
return {
    "action": "request_file",
    "path": path,
    "function_name": function_name,
    "new_code": new_code,
    "original_tc": tc
}
```

### 2. Proxy.py обрабатывает действие

`proxy.py` получает это действие, сохраняет состояние и отправляет клиенту `read_file`:

```python
# В proxy.py, внутри цикла обработки:
if action == "request_file":
    read_file_tc = { ... tool call ... }
    answer.tool_calls = [read_file_tc]
    # Сохраняем контекст для следующей итерации
    answer_processor._pending_function_replace[path] = {
        "function_name": fn_name,
        "new_code": new_code,
        "request_body": current_body
    }
```

### 3. Проблема текущей реализации

Когда клиент возвращает результат `read_file`, `proxy.py` передаёт тело **текущего запроса** в `process()`, но:
- `process()` пытается извлечь `file_lines` только из `_request_body` (текущий запрос)
- `process()` не знает о `_pending_function_replace`, сохранённом в `proxy.py`
- При невозможности чтения файла `process()` снова возвращает `action="request_file"`

**Итог:** Возникает цикл, где `process()` запрашивает файл, `proxy.py` отправляет `read_file`, клиент возвращает результат, но `process()` не может его использовать для преобразования.

## Функция извлечения контента из read_file

### Назначение

Функция `_extract_file_content_from_read_file_calls()` (в `proxy.py`) извлекает содержимое файла из tool calls типа `read_file` в ответе от клиента. Эта функция является ключевым звеном в цепочке передачи данных обратно в `process()`.

### Сигнатура

```python
def _extract_file_content_from_read_file_calls(
    answer: Request,
    target_path: str
) -> Optional[Dict[str, str]]:
```

### Аргументы

- `request`: Объект `Request` с запросом от клиента, содержащий ответы `tool`
- `target_path`: Путь к файлу, для которого нужен контент

### Возвращаемое значение

- `None` — если tool calls отсутствуют или подходящий `read_file` не найден
- `Dict[str, str]` при успехе
```python
data = {
    "type": "file_content",      # Тип данных
    "path": "src/file.py",       # Путь к файлу
    "line_count": 42,            # Общее количество строк в файле
    "complete": true,            # True если файл передан полностью
    "content": {
        "1": "line1",            # Номер строки (строка) -> содержимое строки
        "2": "line2",            # (начиная с "1")
        "3": "line3",
        ...
    }
}
```

### Алгоритм работы

2. Итерирует по всем tool calls в обратном порядке (от конца к началу)
3. Для каждого проверяет `function.name == "read_file"`
4. Извлекает аргументы (`arguments`), ожидая словарь с ключами `path` и `content`
5. Сверяет `path` с `target_path` (если указан)
6. **Определяет полноту файла:**
   - Если `offset=1` и `content` непустой — считает файл полным (`complete=True`)
   - Если `offset>1` — сравнивает количество полученных строк с ожидаемым диапазоном,
     устанавливает `complete=False` если файл прочитан не полностью
   - Записывает `line_count` как общее количество строк в файле (если доступно)
7. Формирует `data` со структурой:
   ```python
   {
       "type": "file_content",
       "path": path,
       "line_count": N,        # общее кол-во строк
       "complete": True/False, # флаг полноты
       "content": {...}        # номер_строки -> текст
   }
   ```
8. Возвращает первое подходящее совпадение

### Функции проверки достаточности данных

Получив `content` файла из результата `read_file`, необходимо проверить, достаточно ли этих данных для завершения операции. Это особенно важно при частичном чтении файла (slice) или когда из большого файла нужна только одна функция.

#### `check_file_sufficiency()`

Проверяет, содержит ли `data` полное содержимое целевого файла.

**Сигнатура:**
```python
def check_file_sufficiency(data: dict, pending_info: dict) -> bool:
```

**Аргументы:**
- `data` — словарь с данными, передаваемый в `process()` (тип `file_content`)
- `pending_info` — информация из `_pending_function_replace` о запрошенном файле

**Возвращает:** `True`, если данные представляют собой полный файл (контент не пуст)

#### `check_function_sufficiency()`

Проверяет, содержит ли `data` полное определение требуемой функции.

**Сигнатура:**
```python
def check_function_sufficiency(data: dict, pending_info: dict) -> bool:
```

**Аргументы:**
- `data` — словарь с данными, передаваемый в `process()`
- `pending_info` — информация из `_pending_function_replace` (содержит `function_name`)

**Логика:**
1. Извлекает `content` из `data`
2. Проверяет, что `content` не пуст
3. Убеждается, что в `content` присутствует полное определение функции (по сигнатуре)

**Возвращает:** `True`, если функция представлена полностью

### Особенности клиентского ответа

Когда `proxy.py` отправляет `read_file`, клиент отвечает тем же tool_call, но с **заполненным** полем `content`:

**Запрос от proxy.py (через LLM):**
```json
{
  "id": "call_req_file_123456",
  "type": "function",
  "function": {
    "name": "read_file",
    "arguments": "{\\"path\\": \\"src/file.py\\", \\"offset\\": 1, \\"mode\\": \\"slice\\"}"
  }
}
```

**Ответ клиента (содержимое файла):**
```json
{
  "id": "call_req_file_123456",
  "type": "function",
  "function": {
    "name": "read_file",
    "arguments": "{\\"path\\": \\"src/file.py\\", \\"content\\": \\"line1\\nline2\\nline3\\"}"
  }
}
```

Функция `_extract_file_content_from_read_file_calls()` как раз извлекает этот `content` по известному `path`.

### Интеграция с data-параметром

После извлечения контента, `proxy.py` должен:

1. Сформировать `data` с полями: `type`, `path`, `line_count`, `complete`, `content`
2. Передать `data` в `process(answer, request_body=current_body, data=data)`

Тогда в `process()`:
- `_request_data` будет содержать `{"type": "file_content", ...}`
- `_extract_content_from_data(path)` вернет список строк файла
- Эти строки будут использованы для преобразования `function_replace` → `apply_diff`

## Формат `data`

Данные передаются в виде словаря с ключами-номерами строк (в виде строк):

```python
data = {
    "type": "file_content",      # Тип данных
    "path": "src/file.py",       # Путь к файлу
    "line_count": 42,            # Общее количество строк в файле
    "complete": true,            # True если файл передан полностью
    "content": {
        "1": "line1",            # Номер строки (строка) -> содержимое строки
        "2": "line2",            # (начиная с "1")
        "3": "line3",
        ...
    }
}
```

`_extract_file_content_from_read_file_calls()` должна устанавливать `complete=True` если количество строк в ответе `read_file` меньше запрошенного (`total_lines` < `offset` + получено строк), либо если `offset=1` и файл получен полностью. `line_count` должна содержать общее количество строк в файле.

Для удобства извлечения строк используется `_extract_content_from_data()`,
который преобразует такой словарь в список строк (отсортированный по номерам).

## Предлагаемое решение

### Архитектура

1. `process()` принимает **необязательный параметр `data`** с дополнительным контекстом
2. `proxy.py` передаёт в `data` результаты `read_file` при повторном вызове `process()`
3. `process()` использует `data` для завершения отложенного преобразования

### Интерфейс `process()`

```python
def process(self, answer, request_body: dict = None, data: dict = None) -> Optional[dict]:
    """
    Args:
        answer: Объект Answer с ответом от LLM
        request_body: Тело исходного запроса
        data: Дополнительные данные, переданные из proxy.py
              Формат: {"type": "file_content", "path": "...", "line_count": N, "complete": T, "content": {...}}
    
    Returns:
        - None: успешно обработано
        - dict: требуется дополнительное действие
    """
```

### Обработка `function_replace` в `process()`

```
1. Получаем path, function_name, new_code из tool_call
2. Пытаемся собрать file_lines:
   a) Из request_body через _extract_file_content_from_request()
   b) Из data[type="file_content"] для целевого path
   c) Прямое чтение файла из workspace
3. Если есть file_lines → конвертируем в apply_diff
4. Если нет file_lines → возвращаем action="request_file"
5. Если есть data с нужным path, но преобразование не удалось → 
   возвращаем ошибку (больше не запрашиваем файл)
```

### Обработка в `proxy.py`

```
1. При получении action="request_file":
   - Сохраняем состояние в _pending_function_replace
   - Отправляем read_file клиенту

2. При получении ответа с read_file (новый запрос от клиента):
   - Извлекаем path и content из tool_call результата
     → используем _extract_file_content_from_read_file_calls()
   - Формируем data = {"type": "file_content", "path": path, "content": content, ...}
   - Передаём data в process(answer, request_body=current_body, data=data)

3. Если process() возвращает None:
   - Преобразование выполнено успешно
   - Продолжаем валидацию tool_calls

4. Если process() снова возвращает action="request_file":
   - Либо файл не найден, либо другая ошибка
   - Отдаём ошибку клиенту (превышен лимит или невозможно выполнить)
```

## Реализация

### Изменения в `answers.py`

1. Добавить параметр `data: dict = None` в сигнатуру `process()`
2. Добавить метод `_extract_content_from_data()` для извлечения file_lines из data
3. В `process_single_tool_call()`, при обработке `function_replace`:
   - Пробуем извлечь file_lines из data перед прямым чтением файла
   - Если есть data с file_content, используем его как приоритетный источник
4. При успешном преобразовании с использованием data — очищать отложенное состояние

### Изменения в `proxy.py`

1. При получении нового запроса от клиента:
   - Проверять `request_body` на наличие `tool_calls` с `name="read_file"`
   - Если есть результат `read_file` для файла из `_pending_function_replace`:
     * Формировать `data` с содержимым файла (используя `_extract_file_content_from_read_file_calls()`)
     * Передавать `data` в `process()`
     * Удалять запись из `_pending_function_replace` после использования

2. При получении `action="request_file"`:
   - Проверять, не было ли уже слишком много попыток
   - Если лимит исчерпан — возвращать ошибку клиенту

3. **Новые функции проверки:**
   - `check_file_sufficiency()` — проверяет полноту файла
   - `check_function_sufficiency()` — проверяет полноту определения функции

## Сценарий работы

### Нормальный сценарий

```
1. LLM возвращает function_replace (нет доступа к файлу)
   ↓
2. process() → returns {"action": "request_file", path, ...}
   ↓
3. proxy.py → сохраняет состояние, отправляет read_file клиенту
   ↓
4. Клиент → выполняет read_file, возвращает результат в новом запросе
   ↓
5. proxy.py → извлекает content из read_file результата
              → формирует data
              → вызывает process(answer, request_body, data)
   ↓
6. process() → извлекает file_lines из data
              → успешно конвертирует в apply_diff
              → возвращает None
   ↓
7. proxy.py → отправляет apply_diff клиенту
```

### Сценарий ошибки (файл не найден)

```
1-4... (то же самое)
   ↓
5. proxy.py → data содержит path, но content пуст/None
              → вызывает process() с data
   ↓
6. process() → проверяет data, видит отсутствие контента
              → возвращает {"action": "request_file", error: "file_not_found"}
   ↓
7. proxy.py → видит повторный запрос файла
              → возвращает ошибку клиенту: "Файл не найден"
```

## Преимущества решения

1. **Чёткий контракт** — `process()` явно запрашивает данные через возвращаемое значение
2. **Локальность** — `process()` остаётся независимым, данные передаются через параметр
3. **Многократный retry** — можно запрашивать разные файлы в цикле
4. **Отказоустойчивость** — корректная обработка ошибок отсутствующих файлов
5. **Тестируемость** — `process()` легко тестировать с передачей data напрямую

## Альтернативные подходы (отклонены)

1. **Глобальное состояние** — хранить все данные в статическом кэше
   ❌ Проблема: утечки памяти, сложность очистки

2. **Передача через request_body** — модифицировать тело запроса
   ❌ Проблема: нарушение формата OpenAI API, путаница

3. **Синхронное чтение** — читать файл прямо в proxy.py до вызова process()
   ❌ Проблема: нарушение разделения ответственности, проблемы с правами доступа

## Тестовые сценарии

1. `test_process_function_replace_with_file_content_data()`
   - Передать function_replace с data[type="file_content"]
   - Проверить успешное преобразование в apply_diff

2. `test_process_function_replace_without_data_retries()`
   - Передать function_replace без данных
   - Проверить возврат action="request_file"

3. `test_proxy_handles_read_file_response()`
   - Сымитировать ответ клиента с read_file
   - Проверить передачу data в process()

4. `test_proxy_timeout_on_missing_file()`
   - Сымитировать цикл запросов при отсутствующем файле
   - Проверить корректную ошибку после лимита

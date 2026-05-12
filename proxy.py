# proxy.py
import time
import json
import uuid
import re
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import os
import atexit
import traceback
from answers import AnswerProcessor, ArgumentParseError
from log import log_response, log_retry_attempt, log_info, log_debug

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from answers import AnswerProcessor
from requests import RequestProcessor


app = FastAPI(title="LiteLLM Proxy — Full Response + Tool Calls Logging + Proxying")

# Настройки
LITELLM_URL = os.getenv("LITELLM_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TIMEOUT = httpx.Timeout(180.0, connect=15.0)

class MaxRetriesExceededError(Exception):
    """Превышено максимальное количество retry при сбоях"""
    pass

def _log_parse_error(error: ArgumentParseError, answer) -> None:
    """
    Логирует ошибку парсинга аргументов tool call.
    """
    log_info("Argument parse error",
              type="argument_parse_error",
              tool_name=error.tool_name,
              tool_call_id=error.tool_call_id,
              error_message=str(error),
              original_args_preview=error.original_args[:500] if error.original_args else None,
              model=answer.model,
              duration=answer.duration,
              is_stream=answer.is_stream,
              status_code=answer.status_code,
              conversation_id=getattr(answer, 'conversation_id', 'unknown'))
    
    # ВЫВОД В КОНСОЛЬ (ГЛАВНОЕ)
    timestamp = datetime.now().isoformat()
    print(f"\n{'='*60}")
    print(f"❌ ОШИБКА ПАРСИНГА АРГУМЕНТОВ")
    print(f"{'='*60}")
    print(f"📅 Время: {timestamp}")
    print(f"🔧 Tool: {error.tool_name}")
    print(f"🆔 Tool Call ID: {error.tool_call_id}")
    print(f"📝 Ошибка: {error}")
    print(f"📄 Аргументы (первые 200 символов):")
    print(f"   {error.original_args[:200] if error.original_args else 'None'}...")
    print(f"🤖 Модель: {answer.model}")
    print(f"⏱️ Длительность: {answer.duration:.2f}с")
    print(f"💬 Conversation ID: {getattr(answer, 'conversation_id', 'unknown')}")
    print(f"{'='*60}\n")


def _log_processing_error(error: Exception, answer, stage: str) -> None:
    """
    Логирует ошибку обработки ответа.
    """
    log_info("Processing error", type="processing_error", stage=stage,
              error_type=type(error).__name__, error_message=str(error),
              traceback=traceback.format_exc() if stage in ["parsing", "validation", "serialization"] else None,
              model=answer.model, duration=answer.duration,
              is_stream=answer.is_stream, status_code=answer.status_code,
              conversation_id=getattr(answer, 'conversation_id', 'unknown'))
    
    # ВЫВОД В КОНСОЛЬ (ГЛАВНОЕ)
    timestamp = datetime.now().isoformat()
    print(f"\n{'='*60}")
    print(f"⚠️ ОШИБКА ОБРАБОТКИ [{stage.upper()}]")
    print(f"{'='*60}")
    print(f"📅 Время: {timestamp}")
    print(f"🔥 Тип ошибки: {type(error).__name__}")
    print(f"📝 Сообщение: {error}")
    print(f"🤖 Модель: {answer.model}")
    print(f"⏱️ Длительность: {answer.duration:.2f}с")
    print(f"💬 Conversation ID: {getattr(answer, 'conversation_id', 'unknown')}")
    
    # Показываем traceback для отладки
    if stage in ["parsing", "validation", "serialization"]:
        print(f"\n📚 Traceback (для отладки):")
        traceback.print_exc()
    
    print(f"{'='*60}\n")


def _log_fatal_error(error: Exception, answer, context: str) -> None:
    """
    Логирует фатальную ошибку.
    """
    log_info("Fatal error", type="fatal_error", context=context,
              error_type=type(error).__name__, error_message=str(error),
              traceback=traceback.format_exc(),
              answer_info={
                  "model": answer.model if answer else 'unknown',
                  "duration": answer.duration if answer else 0,
                  "status_code": answer.status_code if answer else 500
              } if answer else None)
    
    # ВЫВОД В КОНСОЛЬ (ГЛАВНОЕ)
    timestamp = datetime.now().isoformat()
    print(f"\n{'='*60}")
    print(f"💥 ФАТАЛЬНАЯ ОШИБКА [{context.upper()}]")
    print(f"{'='*60}")
    print(f"📅 Время: {timestamp}")
    print(f"🔥 Тип ошибки: {type(error).__name__}")
    print(f"📝 Сообщение: {error}")
    if answer:
        print(f"🤖 Модель: {answer.model}")
        print(f"⏱️ Длительность: {answer.duration:.2f}с")
    print(f"\n📚 Полный traceback:")
    traceback.print_exc()
    print(f"{'='*60}\n")

def _extract_data_from_request(
    pending_info: Dict[str, Any],
    request_body: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Извлекает запрошенные данные (файл или функцию) из тела запроса клиента.
    
    Args:
        pending_info: Информация о том, что ожидается (action, path, function_name)
        request_body: Тело запроса от клиента
        
    Returns:
        Извлеченные данные (file_content или function_content), если найдены и полные.
        None, если данных нет или они неполные.
    """
    if not pending_info:
        return None
    
    path = pending_info.get("path")
    if not path:
        print(f"[DEBUG _extract_data_from_request] Нет path в pending_info")
        return None
    
    action = pending_info.get("action")
    if not action:
        print(f"[DEBUG _extract_data_from_request] Нет action в pending_info")
        return None
    
    print(f"[DEBUG _extract_data_from_request] Извлекаем данные для {action}: path={path}")
    
    # Извлекаем сырой контент файла из запроса
    extracted = _extract_file_content_from_request(request_body, path)
    if extracted is None:
        print(f"[DEBUG _extract_data_from_request] Не удалось извлечь содержимое файла '{path}'")
        return None
    
    print(f"[DEBUG _extract_data_from_request] Получены сырые данные для файла '{path}'")
    
    # В зависимости от действия проверяем полноту данных
    if action == "request_file":
        # Для файла: проверяем целостность (непрерывность от 1 до N, EOF)
        ready_data = check_file_sufficiency(extracted, pending_info)
        if ready_data:
            print(f"✓ Файл '{path}' найден и полный")
            return ready_data
        else:
            print(f"⚠️ Файл '{path}' неполный или отсутствует в запросе")
            return None
    
    elif action == "request_function":
        # Для функции: проверяем наличие полного определения
        function_name = pending_info.get("function_name")
        if not function_name:
            print(f"[DEBUG _extract_data_from_request] Нет function_name в pending_info")
            return None
        
        ready_data = check_function_sufficiency(extracted, function_name)
        if ready_data:
            print(f"✓ Функция '{function_name}()' в '{path}' найдена и полная")
            return ready_data
        else:
            print(f"⚠️ Функция '{function_name}()' в '{path}' неполная или отсутствует")
            return None
    
    else:
        print(f"[DEBUG _extract_data_from_request] Неизвестное действие: {action}")
        return None

def _extract_file_content_from_request(
    body: dict,
    target_path: str
) -> Optional[Dict[str, Any]]:
    """
    Извлекает содержимое файла из read_file tool calls в ЗАПРОСЕ (body).
    """
    if not isinstance(body, dict):
        return None
    
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return None
    
    # Шаг 1: Собираем все read_file запросы
    read_file_requests = {}
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        if msg.get("role") != "assistant":
            continue
        
        tool_calls = msg.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue
        
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            
            func = tc.get('function', {})
            if not isinstance(func, dict):
                continue
            
            if func.get('name') != 'read_file':
                continue
            
            tool_call_id = tc.get('id')
            if not tool_call_id:
                continue
            
            args = func.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if not isinstance(args, dict):
                continue
            
            path = args.get('path')
            if not path:
                continue
            
            offset = args.get('offset', 1)
            limit = args.get('limit', 2000)  # ТОЛЬКО ЗДЕСЬ DEFAULT 2000
            
            read_file_requests[tool_call_id] = {
                "path": path,
                "offset": offset,
                "limit": limit
            }
    
    # Шаг 2: Собираем ответы в обратном порядке
    content_dict = {}
    last_line_num = 0
    total_lines = None
    EOF = False
    
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        
        # Проверка на изменение файла
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    
                    func = tc.get('function', {})
                    if not isinstance(func, dict):
                        continue
                    
                    tool_name = func.get('name', '')
                    if tool_name not in ['apply_diff', 'write_to_file']:
                        continue
                    
                    args = func.get('arguments', {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    
                    if not isinstance(args, dict):
                        continue
                    
                    if args.get('path') == target_path:
                        break
        
        if msg.get("role") != "tool":
            continue
        
        tool_call_id = msg.get("tool_call_id")
        content = msg.get("content")
        
        if not tool_call_id or not content:
            continue
        
        request_info = read_file_requests.get(tool_call_id)
        if not request_info:
            continue
        
        if request_info["path"] != target_path:
            continue
        
        # Парсим строки
        lines = content.split('\n')
        
        # Первая строка - мета-информация
        meta_line = lines[0] if lines else ""
        
        start_shown = None
        end_shown = None
        
        # line_count будет вычисляться в конце на основе всех собранных данных
        
        # Извлекаем показываемый диапазон
        range_match = re.search(r'Showing lines (\d+)-(\d+)', meta_line, re.IGNORECASE)
        if range_match:
            start_shown = int(range_match.group(1))
            end_shown = int(range_match.group(2))
            current_offset = start_shown
            current_limit = end_shown - start_shown + 1
        
        # Определяем EOF:
        # 1. Если есть явный маркер "truncated" - точно не EOF
        # 2. Если end_shown == total_lines - достигнут конец файла
        # 3. Если запрошенный limit > полученных строк - значит файл кончился
        
        if "File content truncated" not in meta_line:
            EOF = True
        
        # Удаляем мета-строки
        lines = lines[1:]  # убираем "File: ..."
        if lines and lines[0].strip().startswith("Status:"):
            lines = lines[1:]  # убираем "Status: ..."
        if lines and lines[0].strip().startswith("IMPORTANT:"):
            lines = lines[1:]  # убираем "IMPORTANT: ..."
            
        # Убираем пустые строки в начале
        while lines and not lines[0].strip():
            lines = lines[0:]
        
        for line in lines:
            if not line.strip():
                continue
            
            # Разбиваем строку по первому символу |
            #print(f"'{line}'")
            parts = line.split('| ', 1)
            #print(parts)
            if len(parts) != 2:
                print(f"[DEBUG _extract_file_content_from_request] Разбиение по | не сработало")
            
            # Левая часть - номер строки
            try:
                line_num = int(parts[0].strip())
            except ValueError:
                continue
            
            # Правая часть - содержимое строки (сохраняем как есть)
            line_content = parts[1]
            # Если строка состоит только из пробелов, заменяем на пустую строку
            if line_content and not line_content.strip():
                line_content = ''
            
            if line_num not in content_dict:
                content_dict[line_num] = line_content
                if line_num > last_line_num:
                    last_line_num = line_num
    
    if content_dict:
        # Вычисляем line_count на основе собранных данных
        line_count = max(int(k) for k in content_dict.keys()) if content_dict else 0
        
        # Сортируем ключи для обеспечения последовательного порядка
        sorted_content = {k: content_dict[k] for k in sorted(content_dict.keys())}
        return {
            'type': 'file_content',
            'path': target_path,
            'content': sorted_content,
            'EOF': EOF,
            'line_count': line_count
        }
    
    return None

def check_file_sufficiency(data: dict, pending_info: dict) -> Optional[dict]:
    """
    Проверяет, содержит ли data полное содержимое целевого файла.
    Возвращает готовый файл для передачи в process(), если данные достаточны.

    Функция проверяет, что data содержит непрерывный и полный контент файла,
    необходимого для операции function_replace.

    Args:
        data: Словарь с данными, передаваемый в process() (тип file_content)
        pending_info: Информация из _process_request о запрошенном файле
                     (содержит path, function_name, full_code)

    Returns:
        Словарь с данными файла, если данные полные и непрерывные.
        None, если данных недостаточно или они неполные.
    """
    # Проверка наличия data
    if not data or not isinstance(data, dict):
        return None
    
    # Проверка типа
    if data.get('type') != 'file_content':
        return None
    
    # Проверка совпадения пути
    data_path = data.get('path')
    pending_path = pending_info.get('path')
    if data_path != pending_path:
        return None
    
    # Проверка контента
    content = data.get('content')
    if not content or not isinstance(content, dict):
        return None
    
    # Проверка EOF
    if not data.get('EOF'):
        return None
    
    # Проверяем, что нет пропусков в нумерации строк
    try:
        line_numbers = sorted([int(k) for k in content.keys()])
    except (ValueError, TypeError):
        return None
    
    if not line_numbers:
        return None
    
    # Проверяем непрерывность от 1 до N
    if line_numbers[0] != 1:
        return None
    
    for i in range(1, len(line_numbers)):
        if line_numbers[i] != line_numbers[i-1] + 1:
            return None

    # Данные достаточны - возвращаем готовый файл
    return {
        "type": "file_content",
        "path": data_path,
        "content": content,
        "line_count": len(line_numbers)
    }

def check_function_sufficiency(data: dict, function_name: str) -> Optional[str]:
    """
    Проверяет, содержит ли data полное определение требуемой функции.
    Возвращает тело функции (включая def) если функция найдена и полная.
    """
    #print(f"[DEBUG check_function_sufficiency] data keys: {data.keys() if data else None}")
    #print(f"[DEBUG check_function_sufficiency] function_name: {function_name}")
    #print(f"[DEBUG check_function_sufficiency] line_count: {data.get('line_count')}")

    if not data or not isinstance(data, dict):
        print(f"[DEBUG check_function_sufficiency] data is None or not dict")
        return None

    content = data.get('content')
    if not content:
        print(f"[DEBUG check_function_sufficiency] No content")
        return None

    # Поиск строки с определением функции
    def_pattern = re.compile(rf'^\s*def\s+{re.escape(function_name)}\s*\(')
    start_line = None
    
    line_count = data.get('line_count')
    
    # Определяем тип ключей в content (int или str)
    sample_key = next(iter(content.keys())) if content else None
    use_int_keys = isinstance(sample_key, int)
    
    for i in range(1, line_count + 1):
        # Используем правильный тип ключа
        key = i if use_int_keys else str(i)
        line_content = content.get(key)
        
        if line_content is None:
            continue
        
        #print(f"[DEBUG check_function_sufficiency] Line {i}: '{line_content[:50]}'")
        
        if def_pattern.search(line_content):
            start_line = i
            #print(f"[DEBUG proxy] Заголовок функции найден на строке {i}")
            break
    
    if start_line is None:
        #print(f"[DEBUG proxy] Заголовок функции '{function_name}' не найден")
        return None
    #else:
        #print(f"[DEBUG proxy] Заголовок функции '{function_name}' найден")
    
    # Определяем базовый отступ
    key = start_line if use_int_keys else str(start_line)
    def_line = content[key]
    base_indent = len(def_line) - len(def_line.lstrip())
    
    # Собираем тело функции
    function_body = {}
    function_body[start_line] = def_line
    
    i = start_line + 1
    while i <= line_count:
        line_content = content[i]
        
        if line_content is None:
            break
        
        line_indent = len(line_content) - len(line_content.lstrip())
        
        # Пустые строки и комментарии - часть тела
        if not line_content.strip() or line_content.strip().startswith('#'):
            function_body[i] = line_content
            i += 1
            continue
        
        # Если встретили строку с отступом <= базового и не декоратор - конец функции
        #print(line_content)
        #print(line_indent)
        if line_indent <= base_indent:
            print(f"[DEBUG proxy] Функция полная (конец на строке {i})")
            #print(function_body)
            return {
                "type": "function_content",
                "path": data["path"],
                "function": function_name,
                "content": function_body
            }
        
        function_body[i] = line_content
        i += 1
    
    print(f"[DEBUG proxy] Функция обрезана (достигнут конец файла)")
    return {
        "type": "function_content",
        "path": data["path"],
        "function": function_name,
        "content": function_body
    }


# Инициализируем процессоры
answer_processor = AnswerProcessor()
request_processor = RequestProcessor()

# Статическая переменная для хранения ожидающих запросов файлов
_process_request_pending: Dict[str, Any] = {}

@dataclass
class Answer:
    """Объект ответа от LLM"""
    full_response: Dict[str, Any]
    status_code: int
    duration: float
    workspace_path: Optional[str] = None
    
    @property
    def content(self) -> str:
        return self.full_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    @content.setter
    def content(self, value: str):
        if "choices" in self.full_response and self.full_response["choices"]:
            if "message" not in self.full_response["choices"][0]:
                self.full_response["choices"][0]["message"] = {}
            self.full_response["choices"][0]["message"]["content"] = value
    
    @property
    def tool_calls(self) -> list:
        return self.full_response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
    
    @tool_calls.setter
    def tool_calls(self, value: list):
        if "choices" in self.full_response and self.full_response["choices"]:
            if "message" not in self.full_response["choices"][0]:
                self.full_response["choices"][0]["message"] = {}
            if value:
                self.full_response["choices"][0]["message"]["tool_calls"] = value
            else:
                self.full_response["choices"][0]["message"].pop("tool_calls", None)
    
    @property
    def model(self) -> str:
        return self.full_response.get("model", "unknown")

def _get_llm_headers(api_key: str) -> Dict[str, str]:
    """
    Формирует заголовки для запроса к LLM.
    
    Args:
        api_key: API ключ для авторизации
        
    Returns:
        Словарь с заголовками
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RooCode Proxy"
    }

async def send_to_llm_with_retry(
    request_body: dict, 
    api_key: str,
    max_retries: int = 2
) -> Dict:
    """
    Отправляет ОДИН И ТОТ ЖЕ запрос с повторами при сбоях.
    
    Args:
        request_body: Тело запроса
        api_key: API ключ для авторизации
        max_retries: Максимальное количество попыток
        
    Returns:
        Словарь с ответом или ошибкой
    """
    last_error = None
    headers = _get_llm_headers(api_key)
    
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    f"{LITELLM_URL}/chat/completions",
                    json=request_body,
                    headers=headers,
                )
                
                duration = time.time() - start_time
                raw_response = await resp.aread()
                
                if resp.status_code >= 500 and attempt < max_retries:
                    # 5xx ошибка - можно повторить тот же запрос
                    continue
                
                if resp.status_code >= 400:
                    return {
                        "full_response": {"error": raw_response.decode('utf-8')},
                        "duration": duration,
                        "status_code": resp.status_code,
                        "is_error": True,
                        "is_stream": False,
                        "headers": dict(resp.headers)
                    }
                
                data = json.loads(raw_response)
                return {
                    "full_response": data,
                    "duration": duration,
                    "status_code": resp.status_code,
                    "is_error": False,
                    "is_stream": False
                }
                
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_error = e
            if attempt == max_retries:
                raise MaxRetriesExceededError(f"Max retries exceeded: {e}")
            continue
    
    raise MaxRetriesExceededError(f"Max retries exceeded: {last_error}")


def create_error_response(
    collected_data: dict,
    is_stream: bool
) -> Response:
    """Создает ответ с ошибкой"""
    error_body = {
        "error": {
            "message": collected_data.get("error", "Unknown error"),
            "type": "api_error",
            "code": collected_data.get("status_code", 500)
        }
    }
    
    return Response(
        content=json.dumps(error_body),
        status_code=collected_data.get("status_code", 500),
        media_type="application/json"
    )


@app.get("/v1/models")
async def list_models():
    """Эндпоинт для получения списка моделей"""
    return {
        "object": "list",
        "data": [
            {
                "id": "openai/gpt-4o",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "openai/gpt-4o-mini",
                "object": "model",
                "created": 1686935000,
                "owned_by": "openai"
            },
            {
                "id": "openai/gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "anthropic/claude-3.5-sonnet",
                "object": "model",
                "created": 1698412800,
                "owned_by": "anthropic"
            },
            {
                "id": "anthropic/claude-3-haiku",
                "object": "model",
                "created": 1698412800,
                "owned_by": "anthropic"
            },
            {
                "id": "google/gemini-pro",
                "object": "model",
                "created": 1700000000,
                "owned_by": "google"
            },
            {
                "id": "google/gemini-flash-1.5",
                "object": "model",
                "created": 1700000000,
                "owned_by": "google"
            },
            {
                "id": "meta-llama/llama-3.1-70b-instruct",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meta"
            },
            {
                "id": "meta-llama/llama-3.1-8b-instruct",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meta"
            }
        ]
    }


@app.get("/models")
async def list_models_alt():
    """Альтернативный эндпоинт для совместимости"""
    return await list_models()


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    global _process_request_pending
    _original_response_text: Optional[str] = None

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON")
    
    # ✅ ПРОЦЕССИРОВАНИЕ ИЗМЕНЕНИЙ ЗАПРОСА (уже логирует внутри)
    modified_body = request_processor.process(body=body.copy())

    # Получаем список доступных инструментов
    available_tools = []
    if "tools" in modified_body:
        available_tools = [tool.get("function", {}).get("name", "") for tool in modified_body["tools"]]
    
    # 📤 Вывод информации о запросе (метаданные) + файл (полный контент)
    print(f"{datetime.now().isoformat()} | REQ | size={len(str(body))} | tools={len(available_tools)}")
    
    # Настройки self-correction
    max_corrections = 2
    current_body = modified_body
    final_answer = None
    correction_attempt = 0

    # Цикл self-correction (отправка НОВЫХ запросов при невалидных tool calls)
    while correction_attempt <= max_corrections:
        pending_data = None
        print(f"[DEBUG proxy] _process_request_pending = {_process_request_pending}")
        if _process_request_pending:
            pending_data = _extract_data_from_request(_process_request_pending,current_body)
            if pending_data is None:
                read_file_response = {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": _process_request_pending["original_response"].model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None,  # Нет текстового контента
                                    "tool_calls": [{
                                        "id": f"call_{uuid.uuid4().hex[:24]}",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": json.dumps({
                                                "path": _process_request_pending.get("path"),
                                                "offset": 1,
                                                "limit": 2000
                                            })
                                        }
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }],
                            "usage":  {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        }
                        
                # Создаем новый Answer объект для read_file
                answer = Answer(
                    full_response=read_file_response,
                    status_code=200,
                    duration=0
                )
                break
            else:
                answer = copy.deepcopy(_process_request_pending["original_response"])
        else:
            # Отправка ТЕКУЩЕГО запроса с retry
            try:
                collected_data = await send_to_llm_with_retry(
                    current_body,
                    OPENROUTER_API_KEY,
                    max_retries=2
                )
            except MaxRetriesExceededError as e:
                # Сетевые сбои - отдаём ошибку клиенту
                print(f"❌ LLM request failed after retries: {e}")
                return Response(
                    content=json.dumps({"error": f"LLM request failed: {str(e)}"}),
                    status_code=504,
                    media_type="application/json"
                )
            
            # Обработка ошибок от LLM
            if collected_data.get("is_error"):
                status_code = collected_data.get('status_code')
                error_detail = collected_data.get('full_response', {}).get('error', '')
                error_headers = collected_data.get('headers', {})
                
                # Краткое описание ошибки из заголовков (не более 150 символов)
                desc_parts = []
                if error_headers.get('x-response-message'):
                    desc_parts.append(error_headers['x-response-message'][:150])
                if error_headers.get('x-error-message'):
                    desc_parts.append(error_headers['x-error-message'][:150])
                if error_headers.get('x-ratelimit-remaining'):
                    desc_parts.append(f"rate_limit={error_headers['x-ratelimit-remaining']}")
                if error_detail:
                    desc_parts.append(str(error_detail)[:150])
                
                desc = ' | '.join(desc_parts)
                if desc:
                    print(f"❌ LLM API error: status={status_code} | {desc}")
                else:
                    print(f"❌ LLM API error: status={status_code}")
                return create_error_response(collected_data, is_stream)
            
            # Создание Answer объекта
            try:
                answer = Answer(
                    full_response=collected_data["full_response"],
                    status_code=collected_data["status_code"],
                    duration=collected_data["duration"]
                )
            except Exception as e:
                _log_fatal_error(e, None, "answer_creation")
                return Response(
                    content=json.dumps({"error": "Internal server error", "details": str(e)}),
                    status_code=500,
                    media_type="application/json"
                )
            
            # Логируем оригинальный ответ через log.py (файл) + консоль (метаданные)
            log_response(
                duration=answer.duration,
                status_code=answer.status_code
            )
            
            # Сохраняем оригинальный ответ в файл немедленно после получения
            save_responce(modified=False, full_responce=answer.full_response)
            tools_list = ",".join([tc.get("function", {}).get("name", "") for tc in answer.tool_calls if "function" in tc])
            print(f"{datetime.now().isoformat()} | ANS | size={len(str(answer.full_response))} | status={answer.status_code} | {tools_list} | duration={answer.duration:.2f}s")
        
        # ✅ ОБРАБОТКА ОТВЕТА (исправление форматирования и т.д.)
        # [DEBUG] Показываем, что передаём в answer_processor.process()
        print(f"[DEBUG proxy] calling answer_processor.process(answer)")
        process_result = answer_processor.process(answer, data=pending_data)

        # Обработка действия от процессора
        if isinstance(process_result, dict):
            action = process_result.get("action")
            func_path = process_result.get("path")
            func_name = process_result.get("function_name")

            if action in ["request_file","request_function"]:
                # Нужно отправить read_file клиенту
                path = process_result.get("path")
                fn_name = process_result.get("function_name")

                _process_request_pending = {
                    "action": action,
                    "path": path,
                    "function_name": fn_name,
                    "original_response": copy.deepcopy(answer),
                    "data": None
                }
                continue
            elif action == "retry_with_llm":
                # Нужно отправить сообщение в LLM для исправления
                retry_message = process_result.get("message")
                # Добавляем в историю и продолжаем цикл
                current_body["messages"].append(retry_message)
                correction_attempt += 1
                continue  # Повторяем запрос к LLM
                
            else:
                # Неизвестное действие - логируем и продолжаем
                print(f"⚠️ Unknown action: {action}")

        # ✅ ПРОВЕРКА СУЩЕСТВОВАНИЯ TOOL CALLS
        is_valid, invalid_tools = answer_processor.validate_tool_calls_exist(
            answer.tool_calls,
            available_tools
        )
        
        if is_valid:
            # Всё хорошо - выходим из цикла
            final_answer = answer
            break
        
        # Невалидные tool calls - пробуем self-correction
        correction_attempt += 1
        
        if correction_attempt > max_corrections:
            # Лимит исчерпан - отдаём ответ как есть (с невалидными tool calls)
            final_answer = answer
            break
        
        current_body = request_processor.build_correction_request(
            current_body,
            invalid_tools,
            available_tools
        )
        
        # Логируем correction запрос через log_modified_request (файл) + консоль (метаданные)
        print(f"{datetime.now().isoformat()} | correction | attempt={correction_attempt} | invalid_tools={len(invalid_tools)}")
        
    # Отправка ответа клиенту
    if final_answer is None:
        final_answer = answer
    
    # Логируем финальный ответ (файл) + консоль (метаданные)
    log_response(
        duration=final_answer.duration,
        status_code=final_answer.status_code
    )
    
    # Сохраняем модифицированный ответ
    save_responce(modified=True, full_responce=final_answer.full_response)
    
    tools_list = ",".join([tc.get("function", {}).get("name", "") for tc in final_answer.tool_calls if "function" in tc])
    print(f"{datetime.now().isoformat()} | ANS | size={len(str(final_answer.full_response))} | status={final_answer.status_code} | {tools_list} | duration={final_answer.duration:.2f}s")
    
    return Response(
        content=json.dumps(final_answer.full_response),
        status_code=final_answer.status_code,
        media_type="application/json"
    )


def save_responce(modified: bool, full_responce: Dict[str, Any]) -> None:
    """Сохраняет ответ в файл responses/original_*.json или responses/modified_*.json"""
    try:
        os.makedirs('responses', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kind = "modified" if modified else "original"
        filename = f"responses/{kind}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(full_responce, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[-] Ошибка при сохранении ответа: {e}")


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("🚀 OpenRouter Proxy Server")
    print("="*60)
    print(f"📡 OpenRouter URL: {LITELLM_URL}")
    print(f"🔑 API Key: {OPENROUTER_API_KEY[:20] if OPENROUTER_API_KEY else 'NOT SET'}...")
    print(f"🌐 Server: http://0.0.0.0:8000")
    print("="*60)
    
    # Запуск сервера
    uvicorn.run(
        "proxy:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False,
        log_level="info"
    )
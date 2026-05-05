# log.py
import json
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

# Конфигурация логов
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Пути к файлам логов
RESPONSE_LOG = os.path.join(LOG_DIR, "llm_responses.log")
REQUEST_LOG = os.path.join(LOG_DIR, "requests.log")
MODIFIED_REQUEST_LOG = os.path.join(LOG_DIR, "modified_requests.log")
TOOL_CALLS_LOG = os.path.join(LOG_DIR, "tool_calls.log")
STATS_LOG = os.path.join(LOG_DIR, "request_stats.log")
ERROR_LOG = os.path.join(LOG_DIR, "errors.log")
RETRY_HISTORY_LOG = os.path.join(LOG_DIR, "retry_history.log")
RETRY_FAILURES_LOG = os.path.join(LOG_DIR, "retry_failures.log")

# Директории для индивидуальных файлов запросов и ответов
REQUESTS_DIR = "requests"
RESPONSES_DIR = "responses"
os.makedirs(REQUESTS_DIR, exist_ok=True)
os.makedirs(RESPONSES_DIR, exist_ok=True)


def log_request(method: str, url: str, headers: dict, body: Optional[dict] = None):
    """
    Логирует входящий запрос.
    
    Args:
        method: HTTP метод
        url: URL запроса
        headers: Заголовки запроса
        body: Тело запроса (объект dict)
    """
    timestamp = datetime.now().isoformat()
    
    # Маскируем敏感ные заголовки
    safe_headers = headers.copy()
    if "authorization" in safe_headers:
        safe_headers["authorization"] = "***"
    if "api-key" in safe_headers:
        safe_headers["api-key"] = "***"
    
    # Подготавливаем тело для логирования - ПОЛНОСТЬЮ БЕЗ ОБРЕЗАНИЯ
    body_preview = None
    if body:
        try:
            # Сериализуем в JSON с отступами для читаемости
            body_preview = json.dumps(body, ensure_ascii=False, indent=2)
        except:
            body_preview = str(body)
    
    log_entry = {
        "timestamp": timestamp,
        "type": "request",
        "method": method,
        "url": url,
        "headers": safe_headers,
        "body_preview": body_preview  # теперь полный, необрезанный JSON
    }
    
    _write_log(REQUEST_LOG, log_entry)


def log_modified_request(method: str, url: str, headers: dict, body: Optional[dict] = None, 
                        modifications: List[str] = None):
    """
    Логирует модифицированный запрос перед отправкой к LLM.
    
    Args:
        method: HTTP метод
        url: URL запроса
        headers: Заголовки запроса
        body: Тело запроса (объект dict)
        modifications: Список сделанных модификаций
    """
    timestamp = datetime.now().isoformat()
    
    # Маскируем敏感ные заголовки
    safe_headers = headers.copy()
    if "authorization" in safe_headers:
        safe_headers["authorization"] = "***"
    if "api-key" in safe_headers:
        safe_headers["api-key"] = "***"

    log_entry = {
        "timestamp": timestamp,
        "type": "modified_request",
        "method": method,
        "url": url,
        "headers": safe_headers,
        "modifications": modifications or [],
        "body_preview": body
    }
    
    _write_log(MODIFIED_REQUEST_LOG, log_entry)


def log_response(model: str, full_response: str, duration: float, 
                 response_type: str = "ORIGINAL", is_stream: bool = False,
                 status_code: int = 200):
    """
    Логирует ответ от LLM.
    
    Args:
        model: Название модели
        full_response: Полный ответ в виде JSON строки
        duration: Длительность запроса в секундах
        response_type: Тип ответа (ORIGINAL/PROCESSED)
        is_stream: Был ли запрос streaming
        status_code: HTTP статус код
    """
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "type": "llm_response",
        "model": model,
        "response_type": response_type,
        "is_stream": is_stream,
        "status_code": status_code,
        "duration": round(duration, 3),
        "response_preview": full_response
    }
    
    _write_log(RESPONSE_LOG, log_entry)
    
    # Также логируем статистику
    log_stats(model, duration, status_code, response_type)


def log_tool_calls(tool_calls: List[Dict]):
    """
    Логирует tool calls из ответа.
    
    Args:
        tool_calls: Список tool calls
    """
    if not tool_calls:
        return
    
    timestamp = datetime.now().isoformat()
    
    for tc in tool_calls:
        # Подготавливаем аргументы для логирования
        arguments = tc.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            try:
                # Пробуем распарсить JSON строку
                arguments = json.loads(arguments)
            except:
                pass  # Оставляем как есть
        
        log_entry = {
            "timestamp": timestamp,
            "type": "tool_call",
            "tool_call_id": tc.get("id", "unknown"),
            "function": tc.get("function", {}).get("name", "unknown"),
            "arguments": arguments
        }
        
        _write_log(TOOL_CALLS_LOG, log_entry)


def log_validation_error(tool_call: Dict, error_details: Dict):
    """
    Логирует ошибку валидации tool call.
    
    Args:
        tool_call: Tool call, вызвавший ошибку
        error_details: Детали ошибки
    """
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "type": "validation_error",
        "tool_call_id": tool_call.get("id", "unknown"),
        "function": tool_call.get("function", {}).get("name", "unknown"),
        "error_details": error_details
    }
    
    _write_log(ERROR_LOG, log_entry)


def log_retry_attempt(conversation_id: str, tool_call_id: str, 
                     attempt: int, errors: List[Dict], 
                     retry_message: Dict) -> None:
    """
    Логирует попытку повторного запроса.
    
    Args:
        conversation_id: ID диалога
        tool_call_id: ID tool call
        attempt: Номер попытки (1-based)
        errors: Список ошибок, вызвавших retry
        retry_message: Сообщение, отправленное в retry
    """
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "type": "retry_attempt",
        "conversation_id": conversation_id,
        "tool_call_id": tool_call_id,
        "attempt": attempt,
        "errors": errors,
        "retry_message": {
            "title": retry_message.get("title"),
            "message": retry_message.get("message"),
            "advice": retry_message.get("advice"),
            "requires_attention": retry_message.get("requires_attention", False)
        }
    }
    
    _write_log(RETRY_HISTORY_LOG, log_entry)


def log_retry_failure(conversation_id: str, tool_call_id: str, 
                     errors: List[Dict], tool_call: Dict, max_retries: int):
    """
    Логирует превышение лимита повторных попыток.
    
    Args:
        conversation_id: ID диалога
        tool_call_id: ID tool call
        errors: Список ошибок
        tool_call: Исходный tool call
        max_retries: Максимальное количество попыток
    """
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "type": "retry_failure",
        "conversation_id": conversation_id,
        "tool_call_id": tool_call_id,
        "error_type": "retry_limit_exceeded",
        "max_retries": max_retries,
        "errors": errors,
        "tool_call": tool_call
    }
    
    _write_log(RETRY_FAILURES_LOG, log_entry)


def log_stats(model: str, duration: float, status_code: int, response_type: str):
    """
    Логирует статистику запросов.
    
    Args:
        model: Название модели
        duration: Длительность запроса
        status_code: HTTP статус код
        response_type: Тип ответа
    """
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "model": model,
        "duration": round(duration, 3),
        "status_code": status_code,
        "response_type": response_type
    }
    
    _write_log(STATS_LOG, log_entry)


def _write_log(filepath: str, data: Dict):
    """
    Внутренняя функция для записи в лог-файл.
    
    Args:
        filepath: Путь к файлу
        data: Данные для записи
    """
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[-] Ошибка записи в лог {filepath}: {e}")


def _write_json_file(filepath: str, data: Dict):
    """
    Записывает данные в JSON файл (полностью перезаписывая файл).
    
    Args:
        filepath: Путь к файлу
        data: Данные для записи
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[-] Ошибка записи в файл {filepath}: {e}")


def save_request_to_file(method: str, url: str, headers: dict, body: Optional[dict] = None, prefix: str = "req") -> str:
    """
    Сохраняет запрос в отдельный JSON файл в директории requests/.
    
    Args:
        method: HTTP метод
        url: URL запроса
        headers: Заголовки запроса
        body: Тело запроса
        prefix: Префикс имени файла (req/req_mod)
        
    Returns:
        Имя созданного файла
    """
    timestamp = datetime.now()
    filename = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_request.json"
    filepath = os.path.join(REQUESTS_DIR, filename)
    
    # Маскируем чувствительные заголовки
    safe_headers = headers.copy()
    if "authorization" in safe_headers:
        safe_headers["authorization"] = "***"
    if "api-key" in safe_headers:
        safe_headers["api-key"] = "***"
    
    data = {
        "timestamp": timestamp.isoformat(),
        "method": method,
        "url": url,
        "headers": safe_headers,
        "body": body
    }
    
    _write_json_file(filepath, data)
    return filename


def save_response_to_file(model: str, full_response: dict, duration: float,
                         response_type: str = "ORIGINAL", is_stream: bool = False,
                         status_code: int = 200, prefix: str = "resp") -> str:
    """
    Сохраняет ответ в отдельный JSON файл в директории responses/.
    
    Args:
        model: Название модели
        full_response: Полный ответ (dict)
        duration: Длительность запроса в секундах
        response_type: Тип ответа (ORIGINAL/PROCESSED)
        is_stream: Был ли запрос streaming
        status_code: HTTP статус код
        prefix: Префикс имени файла
        
    Returns:
        Имя созданного файла
    """
    timestamp = datetime.now()
    filename = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_response.json"
    filepath = os.path.join(RESPONSES_DIR, filename)
    
    data = {
        "timestamp": timestamp.isoformat(),
        "model": model,
        "response_type": response_type,
        "is_stream": is_stream,
        "status_code": status_code,
        "duration": round(duration, 3),
        "response": full_response
    }
    
    _write_json_file(filepath, data)
    return filename



def cleanup_logs(backup: bool = True):
    """
    Очищает лог-файлы, создавая бэкапы.
    
    Args:
        backup: Создавать ли бэкапы
    """
    log_files = [RESPONSE_LOG, REQUEST_LOG, MODIFIED_REQUEST_LOG, 
                 TOOL_CALLS_LOG, STATS_LOG, ERROR_LOG, 
                 RETRY_HISTORY_LOG, RETRY_FAILURES_LOG]
    
    archive_dir = os.path.join(LOG_DIR, "archive")
    if backup:
        os.makedirs(archive_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n🧹 Очистка лог-файлов при запуске:")
    
    for log_file in log_files:
        if os.path.exists(log_file):
            if backup:
                # Создаем бэкап
                backup_name = os.path.basename(log_file) + f".{timestamp}.bak"
                backup_path = os.path.join(archive_dir, backup_name)
                try:
                    with open(log_file, "r", encoding="utf-8") as src:
                        with open(backup_path, "w", encoding="utf-8") as dst:
                            dst.write(src.read())
                    print(f"  📦 Создан бэкап: {backup_name}")
                except Exception as e:
                    print(f"  ❌ Ошибка создания бэкапа {log_file}: {e}")
            
            # Очищаем файл
            try:
                open(log_file, "w").close()
                print(f"  🧹 Очищен: {os.path.basename(log_file)}")
            except Exception as e:
                print(f"  ❌ Ошибка очистки {log_file}: {e}")
    
    print("  ✅ Логи очищены, бэкапы в", archive_dir)


def check_log_files():
    """Проверяет существование всех лог-файлов и создает их при необходимости"""
    log_files = [RESPONSE_LOG, REQUEST_LOG, MODIFIED_REQUEST_LOG, 
                 TOOL_CALLS_LOG, STATS_LOG, ERROR_LOG, 
                 RETRY_HISTORY_LOG, RETRY_FAILURES_LOG]
    
    print(f"\n📁 Директория логов существует: {LOG_DIR}")
    print("\n📄 Проверка лог-файлов:")
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("")
                print(f"  ✅ Создан: {os.path.basename(log_file)}")
            except Exception as e:
                print(f"  ❌ Ошибка создания {os.path.basename(log_file)}: {e}")
        else:
            print(f"  ✅ Файл существует: {os.path.basename(log_file)}")


def get_logs_summary(hours: int = 24) -> Dict:
    """
    Возвращает сводку по логам за последние N часов.
    
    Args:
        hours: Количество часов для анализа
        
    Returns:
        Dict со сводкой
    """
    import time
    from collections import Counter
    
    summary = {
        "total_requests": 0,
        "modified_requests": 0,
        "successful": 0,
        "failed": 0,
        "retry_attempts": 0,
        "retry_failures": 0,
        "validation_errors": 0,
        "models": Counter(),
        "avg_duration": 0
    }
    
    cutoff_time = time.time() - (hours * 3600)
    total_duration = 0
    duration_count = 0
    
    # Анализ статистики
    if os.path.exists(STATS_LOG):
        try:
            with open(STATS_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                        
                        if entry_time >= cutoff_time:
                            summary["total_requests"] += 1
                            summary["models"][entry["model"]] += 1
                            
                            if entry["status_code"] == 200:
                                summary["successful"] += 1
                            else:
                                summary["failed"] += 1
                            
                            total_duration += entry["duration"]
                            duration_count += 1
                    except:
                        continue
        except:
            pass
    
    # Анализ модифицированных запросов
    if os.path.exists(MODIFIED_REQUEST_LOG):
        try:
            with open(MODIFIED_REQUEST_LOG, "r", encoding="utf-8") as f:
                summary["modified_requests"] = sum(1 for _ in f)
        except:
            pass
    
    # Анализ retry истории
    if os.path.exists(RETRY_HISTORY_LOG):
        try:
            with open(RETRY_HISTORY_LOG, "r", encoding="utf-8") as f:
                summary["retry_attempts"] = sum(1 for _ in f)
        except:
            pass
    
    # Анализ retry failures
    if os.path.exists(RETRY_FAILURES_LOG):
        try:
            with open(RETRY_FAILURES_LOG, "r", encoding="utf-8") as f:
                summary["retry_failures"] = sum(1 for _ in f)
        except:
            pass
    
    # Анализ ошибок валидации
    if os.path.exists(ERROR_LOG):
        try:
            with open(ERROR_LOG, "r", encoding="utf-8") as f:
                summary["validation_errors"] = sum(1 for _ in f)
        except:
            pass
    
    if duration_count > 0:
        summary["avg_duration"] = total_duration / duration_count
    
    summary["models"] = dict(summary["models"])
    
    return summary
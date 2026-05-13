# log.py
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Конфигурация логов
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Основной лог-файл
MAIN_LOG = os.path.join(LOG_DIR, "main.log")


def log_info(message: str, **kwargs):
    """
    Записывает информационное сообщение в main.log и консоль.
    
    Args:
        message: Текст сообщения
        **kwargs: Дополнительные поля для лога
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": message,
        **kwargs
    }
    _write_log(entry)
    # Выводим Also в консоль
    logger.info(message)


def log_debug(message: str, **kwargs):
    """
    Записывает отладочное сообщение в main.log и консоль.
    
    Args:
        message: Текст сообщения
        **kwargs: Дополнительные поля для лога
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": "DEBUG",
        "message": message,
        **kwargs
    }
    _write_log(entry)
    # Выводим Also в консоль
    logger.debug(message)


def log_modified_request(method: str, url: str, headers: dict,
                         modifications: List[str] = None):
    """
    Логирует модифицированный запрос (без тела) перед отправкой к LLM.
    
    Args:
        method: HTTP метод
        url: URL запроса
        headers: Заголовки запроса
        modifications: Список сделанных модификаций
    """
    # Маскируем чувствительные заголовки
    safe_headers = headers.copy()
    if "authorization" in safe_headers:
        safe_headers["authorization"] = "***"
    if "api-key" in safe_headers:
        safe_headers["api-key"] = "***"
    
    log_info("Modified request sent", type="modified_request", method=method, url=url,
             headers=safe_headers, modifications=modifications or [])


def log_response(duration: float, status_code: int = 200):
    """
    Логирует метаданные ответа (без тела).
    
    Args:
        duration: Длительность запроса в секундах
        status_code: HTTP статус код
    """
    log_info("Response received", type="response", duration=duration, status_code=status_code)


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
    log_info("Retry attempt", type="retry_attempt", conversation_id=conversation_id,
             tool_call_id=tool_call_id, attempt=attempt, errors=errors,
             retry_message={
                 "title": retry_message.get("title"),
                 "message": retry_message.get("message"),
                 "advice": retry_message.get("advice"),
                 "requires_attention": retry_message.get("requires_attention", False)
             })


def log_stats(model: str, duration: float, status_code: int, response_type: str):
    """
    Логирует статистику запросов.
    
    Args:
        model: Название модели
        duration: Длительность запроса
        status_code: HTTP статус код
        response_type: Тип ответа
    """
    log_debug("Request stats", type="stats", model=model, duration=round(duration, 3),
              status_code=status_code, response_type=response_type)


def _write_log(data: Dict):
    """
    Внутренняя функция для записи в main.log.
    
    Args:
        data: Данные для записи
    """
    try:
        with open(MAIN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Ошибка записи в лог {MAIN_LOG}: {e}")

# proxy.py - OpenRouter Proxy with Authorization
# Передаёт запросы и ответы с авторизацией

import logging
import json
import copy
import os
import time
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from answers import AnswerProcessor
from requests import RequestProcessor, _extract_data_from_request
from handlers import save_json_content
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Настройка логгера
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler('logs/test_proxy.log', mode='a', encoding='utf-8')  # Запись в файл
    ]
)
logger = logging.getLogger("proxy")

# Отключаем логирование httpx и httpcore (HTTP Request/Response сообщения)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Инициализируем процессоры
answer_processor = AnswerProcessor()
request_processor = RequestProcessor()

# Статическая переменная для хранения ожидающих запросов файлов
_process_request_pending: Dict[str, Any] = {}

app = FastAPI(title="OpenRouter Proxy")


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

# Целевой URL (куда проксировать запросы)
load_dotenv()
OPENROUTER_URL = os.getenv("OPENROUTER_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _get_llm_headers(api_key: str) -> dict:
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
        "X-Title": "LLMProxy"
    }


async def send_with_retry(
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
    headers = _get_llm_headers(api_key)
    url = f"{OPENROUTER_URL}/chat/completions"
    
    for attempt in range(max_retries + 1):
        logger.debug(f"[DEBUG send_with_retry] Начало цикла retry attempt={attempt + 1}/{max_retries + 1}")
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    url,
                    json=request_body,
                    headers=headers,
                )
                
                duration = time.time() - start_time
                raw_response = await resp.aread()
                
                if resp.status_code >= 500 and attempt < max_retries:
                    # 5xx ошибка - можно повторить тот же запрос
                    logger.warning(f"Retry attempt {attempt + 1}/{max_retries} due to 5xx error: {resp.status_code}")
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
            if attempt == max_retries:
                return {
                    "full_response": {"error": f"Max retries exceeded: {str(e)}"},
                    "duration": 0,
                    "status_code": 504,
                    "is_error": True,
                    "is_stream": False
                }
            logger.warning(f"Retry attempt {attempt + 1}/{max_retries} due to network error: {e}")
            continue
    
    return {
        "full_response": {"error": "Unknown error"},
        "duration": 0,
        "status_code": 500,
        "is_error": True,
        "is_stream": False
    }


def save_response(modified: bool, full_response: Dict[str, Any]) -> None:
    """Сохраняет ответ в файл responses/original_*.json или responses/modified_*.json"""
    prefix = "modified_response" if modified else "original_response"
    save_json_content("responses", full_response, prefix)


def create_error_response(
    collected_data: dict,
    is_stream: bool
) -> Response:
    """Создает ответ с ошибкой"""
    error_body = {
        "error": {
            "message": collected_data.get("full_response", {}).get("error", "Unknown error"),
            "type": "api_error",
            "code": collected_data.get("status_code", 500)
        }
    }
    
    return Response(
        content=json.dumps(error_body),
        status_code=collected_data.get("status_code", 500),
        media_type="application/json"
    )


def _log_fatal_error(error: Exception, answer, context: str) -> None:
    """
    Логирует фатальную ошибку.
    """
    import traceback
    logger.error(f"Fatal error in {context}: {type(error).__name__}: {error}")
    logger.error(traceback.format_exc())


@app.get("/v1/models")
async def list_models():
    """Эндпоинт для получения списка моделей от OpenRouter (только бесплатные)"""
    try:
        headers = _get_llm_headers(OPENROUTER_API_KEY)
        url = f"{OPENROUTER_URL}/models"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            
        if resp.status_code != 200:
            logger.error(f"Failed to fetch models from OpenRouter: {resp.status_code}")
            return {"object": "list", "data": []}
            
        data = resp.json()
        all_models = data.get("data", [])
        
        # Фильтруем только бесплатные модели
        free_models = []
        for model in all_models:
            pricing = model.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0) or 0)
            completion_price = float(pricing.get("completion", 0) or 0)
            
            # Бесплатная модель - если оба цены равны 0
            if prompt_price == 0 and completion_price == 0:
                free_models.append({
                    "id": model.get("id"),
                    "object": "model",
                    "created": model.get("created", 0),
                    "owned_by": model.get("owned_by", "unknown"),
                    "name": model.get("name", ""),
                    "description": model.get("description", "")
                })
        
        return {
            "object": "list",
            "data": free_models
        }
    except Exception as e:
        logger.error(f"Error fetching models from OpenRouter: {e}")
        return {"object": "list", "data": []}


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
    logger.info(f"REQ | size={len(str(body))} | tools={len(available_tools)}")
    
    # Настройки self-correction
    max_corrections = 2
    current_body = modified_body
    final_answer = None
    correction_attempt = 0

    # Цикл self-correction (отправка НОВЫХ запросов при невалидных tool calls)
    while correction_attempt <= max_corrections:
        logger.debug(f"[DEBUG proxy_chat_completions] Начало цикла while correction_attempt={correction_attempt}/{max_corrections}")
        pending_data = None
        if _process_request_pending:
            logger.debug(f"[DEBUG proxy_chat_completions] Ветка if: _process_request_pending существует, path={_process_request_pending.get('path')}")
            pending_data = _extract_data_from_request(_process_request_pending, current_body)
            if pending_data is None:
                logger.debug(f"[DEBUG proxy_chat_completions] Ветка if (nested): pending_data is None - возвращаем read_file")
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
                logger.debug(f"[DEBUG proxy_chat_completions] Ветка else (nested): pending_data найден, используем deep copy original_response")
                answer = copy.deepcopy(_process_request_pending["original_response"])
                logger.info(f"REQ | Получены данные для файла: path={pending_data.get('path')}")
                # Обрабатываем ответ с данными файла (apply_diff, function_replace)
        else:
            logger.debug(f"[DEBUG proxy_chat_completions] Ветка else: _process_request_pending пуст, отправляем запрос в LLM")
            # Отправка ТЕКУЩЕГО запроса с retry
            collected_data = await send_with_retry(
                current_body,
                OPENROUTER_API_KEY,
                max_retries=2
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
                    logger.error(f"❌ LLM API error: status={status_code} | {desc}")
                else:
                    logger.error(f"❌ LLM API error: status={status_code}")
                return create_error_response(collected_data, False)
            
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
            #log_response(
            #    duration=answer.duration,
            #    status_code=answer.status_code
            #)
            
            # Сохраняем оригинальный ответ в файл немедленно после получения
            save_response(modified=False, full_response=answer.full_response)
            tools_list = ",".join([tc.get("function", {}).get("name", "") for tc in answer.tool_calls if "function" in tc])
            logger.info(f"ANS | size={len(str(answer.full_response))} | status={answer.status_code} | {tools_list} | duration={answer.duration:.2f}s")
        
        # ✅ ОБРАБОТКА ОТВЕТА (исправление форматирования и т.д.)
        process_result = answer_processor.process(answer, data=pending_data)

        # Обработка действия от процессора
        if isinstance(process_result, dict):
            action = process_result.get("action")
            func_path = process_result.get("path")
            func_name = process_result.get("function_name")

            if action in ["request_file","request_function"]:
                logger.debug(f"[DEBUG proxy_chat_completions] Ветка if: action in [request_file, request_function]")
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
                logger.debug(f"[DEBUG proxy_chat_completions] Ветка elif: action == retry_with_llm")
                # Нужно отправить сообщение в LLM для исправления
                retry_message = process_result.get("message")
                # Добавляем в историю и продолжаем цикл
                current_body["messages"].append(retry_message)
                correction_attempt += 1
                continue  # Повторяем запрос к LLM
                
            else:
                logger.debug(f"[DEBUG proxy_chat_completions] Ветка else: Unknown action={action}")
                # Неизвестное действие - логируем и продолжаем
                logger.warning(f"⚠️ Unknown action: {action}")

        # ✅ ПРОВЕРКА СУЩЕСТВОВАНИЯ TOOL CALLS
        is_valid, invalid_tools = answer_processor.validate_tool_calls_exist(
            answer.tool_calls,
            available_tools
        )
        
        if is_valid:
            logger.debug(f"[DEBUG proxy_chat_completions] Ветка if: is_valid=True - выходим из цикла")
            # Всё хорошо - выходим из цикла
            final_answer = answer
            break
        
        # Невалидные tool calls - пробуем self-correction
        correction_attempt += 1
        
        if correction_attempt > max_corrections:
            logger.debug(f"[DEBUG proxy_chat_completions] Ветка if: correction_attempt > max_corrections - выходим с невалидными tool calls")
            # Лимит исчерпан - отдаём ответ как есть (с невалидными tool calls)
            final_answer = answer
            break
        
        current_body = request_processor.build_correction_request(
            current_body,
            invalid_tools,
            available_tools
        )
        
        # Логируем correction запрос через log_modified_request (файл) + консоль (метаданные)
        logger.info(f"correction | attempt={correction_attempt} | invalid_tools={len(invalid_tools)}")
        
    # Отправка ответа клиенту
    if final_answer is None:
        final_answer = answer
    
    # Логируем финальный ответ (файл) + консоль (метаданные)
    #log_response(
    #    duration=final_answer.duration,
    #    status_code=final_answer.status_code
    #)
    
    # Сохраняем модифицированный ответ
    save_response(modified=True, full_response=final_answer.full_response)
    
    tools_list = ",".join([tc.get("function", {}).get("name", "") for tc in final_answer.tool_calls if "function" in tc])
    logger.info(f"ANS | size={len(str(final_answer.full_response))} | status={final_answer.status_code} | {tools_list} | duration={final_answer.duration:.2f}s")
    
    return Response(
        content=json.dumps(final_answer.full_response),
        status_code=final_answer.status_code,
        media_type="application/json"
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 50)
    logger.info("🚀 OpenRouter Proxy")
    logger.info(f"Target: {OPENROUTER_URL}")
    logger.info("=" * 50)
    
    uvicorn.run(
        "proxy:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False,
        log_level="info"
    )
# proxy.py
import time
import json
import uuid
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
from log import log_request, log_modified_request, log_response, log_tool_calls, log_validation_error, log_retry_attempt, log_stats, cleanup_logs, check_log_files, get_logs_summary, save_request_to_file, save_response_to_file

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
print(f"4. LITELLM_URL={LITELLM_URL}, API_KEY={'SET' if OPENROUTER_API_KEY else 'NOT SET'}")

class MaxRetriesExceededError(Exception):
    """Превышено максимальное количество retry при сбоях"""
    pass

def _log_parse_error(error: ArgumentParseError, answer) -> None:
    """
    Логирует ошибку парсинга аргументов tool call.
    """
    timestamp = datetime.now().isoformat()
    
    # ВЫВОД В КОНСОЛЬ (ГЛАВНОЕ)
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
    
    # Запись в файл
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_entry = {
        "timestamp": timestamp,
        "type": "argument_parse_error",
        "tool_name": error.tool_name,
        "tool_call_id": error.tool_call_id,
        "error_message": str(error),
        "original_args_preview": error.original_args[:500] if error.original_args else None,
        "model": answer.model,
        "duration": answer.duration,
        "is_stream": answer.is_stream,
        "status_code": answer.status_code,
        "conversation_id": getattr(answer, 'conversation_id', 'unknown')
    }
    
    parse_errors_log = os.path.join(logs_dir, "argument_parse_errors.log")
    try:
        with open(parse_errors_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[-] Ошибка записи в файл: {e}")


def _log_processing_error(error: Exception, answer, stage: str) -> None:
    """
    Логирует ошибку обработки ответа.
    """
    timestamp = datetime.now().isoformat()
    
    # ВЫВОД В КОНСОЛЬ (ГЛАВНОЕ)
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
    
    # Запись в файл
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_entry = {
        "timestamp": timestamp,
        "type": "processing_error",
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "model": answer.model,
        "duration": answer.duration,
        "is_stream": answer.is_stream,
        "status_code": answer.status_code,
        "conversation_id": getattr(answer, 'conversation_id', 'unknown')
    }
    
    processing_errors_log = os.path.join(logs_dir, "processing_errors.log")
    try:
        with open(processing_errors_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[-] Ошибка записи в файл: {e}")


def _log_fatal_error(error: Exception, answer, context: str) -> None:
    """
    Логирует фатальную ошибку.
    """
    timestamp = datetime.now().isoformat()
    
    # ВЫВОД В КОНСОЛЬ (ГЛАВНОЕ)
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
    
    # Запись в файл
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_entry = {
        "timestamp": timestamp,
        "type": "fatal_error",
        "context": context,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "answer_info": {
            "model": answer.model if answer else 'unknown',
            "duration": answer.duration if answer else 0,
            "status_code": answer.status_code if answer else 500
        } if answer else None
    }
    
    fatal_errors_log = os.path.join(logs_dir, "fatal_errors.log")
    try:
        with open(fatal_errors_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[-] Ошибка записи в файл: {e}")


# Инициализируем процессоры
try:
    answer_processor = AnswerProcessor()
    request_processor = RequestProcessor()
except Exception as e:
    print(f"Warning: Could not initialize processors: {e}")
    # Создаем заглушки
    class DummyProcessor:
        def process(self, *args, **kwargs):
            return args[0] if args else {}
        changed = False
        changes_log = []
    
    answer_processor = DummyProcessor()
    request_processor = DummyProcessor()

@dataclass
class Answer:
    """Объект ответа от LLM"""
    full_response: Dict[str, Any]
    is_stream: bool
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


async def send_to_llm_with_retry(
    request_body: dict, 
    headers: dict, 
    max_retries: int = 2
) -> Dict:
    """
    Отправляет ОДИН И ТОТ ЖЕ запрос с повторами при сбоях.
    """
    last_error = None
    
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
                        "is_stream": False
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
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON")

    # Подготовка заголовков
    headers = {
        "Content-Type": "application/json",
    }
    
    # Используем OpenRouter API ключ
    if not OPENROUTER_API_KEY:
        print(f"⚠️ ERROR: OPENROUTER_API_KEY not set!")
        return Response(
            content=json.dumps({"error": "API key not configured"}),
            status_code=500,
            media_type="application/json"
        )
    
    headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
    headers["HTTP-Referer"] = "http://localhost:8000"
    headers["X-Title"] = "RooCode Proxy"
    
    model = body.get("model", "unknown")
    is_stream = body.get("stream", False)
    
    # ✅ ПРОЦЕССИРОВАНИЕ ИЗМЕНЕНИЙ ЗАПРОСА (уже логирует внутри)
    modified_body = request_processor.process(
        body=body.copy(),
        method=request.method,
        url=str(request.url),
        headers=headers
    )
    
    # ✅ СОХРАНЕНИЕ ИСХОДНОГО ЗАПРОСА В ФАЙЛ
    try:
        save_request_to_file(
            method=request.method,
            url=str(request.url),
            headers=headers,
            body=body,
            prefix="req"
        )
    except Exception as e:
        print(f"⚠️ Ошибка сохранения запроса в файл: {e}")
    
    # Проверяем, были ли изменения
    request_changed = getattr(request_processor, 'changed', False)
    print(f"{datetime.now().isoformat()} | request_modified | changed={request_changed}")

    # Получаем список доступных инструментов
    available_tools = []
    if "tools" in modified_body:
        available_tools = [tool.get("function", {}).get("name", "") for tool in modified_body["tools"]]
    
    # Настройки self-correction
    max_corrections = 2
    current_body = modified_body
    final_answer = None
    correction_attempt = 0
    
    # Цикл self-correction (отправка НОВЫХ запросов при невалидных tool calls)
    while correction_attempt <= max_corrections:
        # Отправка ТЕКУЩЕГО запроса с retry (повторы при сбоях)
        try:
            collected_data = await send_to_llm_with_retry(
                current_body,
                headers,
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
        
        # Обработка ошибок от LLM (не сетевых, а API ошибок)
        if collected_data.get("is_error"):
            print(f"❌ LLM API error: status={collected_data.get('status_code')}")
            return create_error_response(collected_data, is_stream)
        
        # Создание Answer объекта
        try:
            answer = Answer(
                full_response=collected_data["full_response"],
                is_stream=is_stream,
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
        
        # Логируем оригинальный ответ через log.py
        log_response(
            model=answer.model,
            full_response=json.dumps(answer.full_response),
            duration=answer.duration,
            response_type="ORIGINAL",
            is_stream=answer.is_stream,
            status_code=answer.status_code
        )
        
        # ✅ СОХРАНЕНИЕ ОРИГИНАЛЬНОГО ОТВЕТА В ФАЙЛ
        try:
            save_response_to_file(
                model=answer.model,
                full_response=answer.full_response,
                duration=answer.duration,
                response_type="ORIGINAL",
                is_stream=answer.is_stream,
                status_code=answer.status_code,
                prefix="resp_orig"
            )
        except Exception as e:
            print(f"⚠️ Ошибка сохранения оригинального ответа в файл: {e}")
        
        # Устанавливаем workspace_path если есть
        if hasattr(request_processor, 'workspace_path'):
            answer.workspace_path = request_processor.workspace_path
        
        # ✅ ОБРАБОТКА ОТВЕТА (исправление форматирования и т.д.)
        try:
            answer_processor.process(answer)
        except ArgumentParseError as e:
            # Ошибка парсинга аргументов - не исправили, отдаём как есть
            _log_parse_error(e, answer)
            final_answer = answer
            break
        except Exception as e:
            _log_processing_error(e, answer, "processing")
            final_answer = answer
            break
        
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
            print(f"⚠️ Max corrections ({max_corrections}) exceeded, returning response with invalid tools")
            final_answer = answer
            break
        
        # Формируем НОВЫЙ запрос с сообщением об ошибке
        print(f"🔄 Self-correction attempt {correction_attempt}/{max_corrections}")
        print(f"   Invalid tools: {[t['tool_name'] for t in invalid_tools]}")
        
        current_body = request_processor.build_correction_request(
            current_body,
            invalid_tools,
            available_tools
        )
        
        # Логируем correction запрос через log_modified_request
        log_modified_request(
            method=request.method,
            url=str(request.url),
            headers=headers,
            body=current_body,
            modifications=[f"self-correction attempt {correction_attempt}: added error message for invalid tools"]
        )
        
        # Продолжаем цикл с новым запросом
    
    # Отправка ответа клиенту
    if final_answer is None:
        final_answer = answer
    
    # Фильтруем невалидные tool calls перед отправкой клиенту
    # (например, если LLM вернул tool calls, которых нет в списке доступных)
    if final_answer.tool_calls and available_tools:
        original_tool_calls = final_answer.tool_calls.copy()
        valid_tool_calls = [
            tc for tc in final_answer.tool_calls
            if tc.get('function', {}).get('name', '') in available_tools
        ]
        
        removed_count = len(original_tool_calls) - len(valid_tool_calls)
        if removed_count > 0:
            final_answer.tool_calls = valid_tool_calls
            print(f"⚠️ Отфильтровано {removed_count} невалидных tool call(s) (недоступны в config/tools.yaml)")
            
            # Обновляем full_response чтобы синхронизировать с tool_calls
            if final_answer.tool_calls:
                final_answer.full_response['choices'][0]['message']['tool_calls'] = final_answer.tool_calls
            else:
                # Удаляем поле tool_calls, если не осталось валидных вызовов
                final_answer.full_response['choices'][0]['message'].pop('tool_calls', None)
    
    # Логируем финальный ответ
    log_response(
        model=final_answer.model,
        full_response=json.dumps(final_answer.full_response),
        duration=final_answer.duration,
        response_type="PROCESSED",
        is_stream=is_stream,
        status_code=final_answer.status_code
    )
    
    # ✅ СОХРАНЕНИЕ ФИНАЛЬНОГО ОТВЕТА В ФАЙЛ
    try:
        save_response_to_file(
            model=final_answer.model,
            full_response=final_answer.full_response,
            duration=final_answer.duration,
            response_type="PROCESSED",
            is_stream=is_stream,
            status_code=final_answer.status_code,
            prefix="resp"
        )
    except Exception as e:
        print(f"⚠️ Ошибка сохранения ответа в файл: {e}")
    
    response_json = json.dumps(final_answer.full_response)
    tool_calls_count = len(final_answer.tool_calls) if hasattr(final_answer, 'tool_calls') else 0
    was_fixed = hasattr(answer_processor, 'changed') and answer_processor.changed
    
    tool_calls_info = ""
    if tool_calls_count > 0:
        tool_calls_info = f" | tool_calls={tool_calls_count}"
        if was_fixed:
            tool_calls_info += " | FIXED"
        else:
            tool_calls_info += " | NOT_FIXED"
    
    if correction_attempt > 0:
        tool_calls_info += f" | corrections={correction_attempt}"
    
    print(f"{datetime.now().isoformat()} | response status={final_answer.status_code} | len={len(response_json)}{tool_calls_info}")
    
    return Response(
        content=json.dumps(final_answer.full_response),
        status_code=final_answer.status_code,
        media_type="application/json"
    )


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
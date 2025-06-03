from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import aiohttp
import uuid
from datetime import datetime
import time
import os
import dotenv

dotenv.load_dotenv()

app = FastAPI()

# OpenAI 标准请求模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "openai-gpt-4.1"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# Notion API 配置
NOTION_API_URL = os.getenv("NOTION_API_URL")

NOTION_HEADERS = {
    "Content-Type": "application/json",
    "Pragma": "no-cache",
    "Accept": "application/x-ndjson",
    "Sec-Fetch-Site": "same-origin",
    "Accept-Language": "zh-CN,zh-Hans;q=0.9",
    "Sec-Fetch-Mode": "cors",
    "Cache-Control": "no-cache",
    "Origin": "https://www.notion.so",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
    "Referer": "https://www.notion.so/chat",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "empty",
    "Cookie": os.getenv("NOTION_COOKIE"),
    "notion-client-version": "23.13.0.3718",
    "x-notion-space-id": os.getenv("NOTION_SPACE_ID"),
    "Priority": "u=3, i",
    "notion-audit-log-platform": "web",
    "x-notion-active-user-header": os.getenv("NOTION_ACTIVE_USER_HEADER")
}

trace_id = os.getenv("TRACE_ID")
space_id = os.getenv("SPACE_ID")
thread_id = os.getenv("THREAD_ID")
user_id = os.getenv("USER_ID")
space_view_id = os.getenv("SPACE_VIEW_ID")

NOTION_BEARER_TOKEN = os.getenv("NOTION_BEARER_TOKEN")

def verify_bearer_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少或无效的 Authorization 头")
    token = authorization.split("Bearer ")[1]
    if token != NOTION_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="无效的 Bearer Token")

def convert_to_notion_format(
    messages: List[ChatMessage],
    context_overrides: dict = {}
) -> Dict[str, Any]:
    """
    更灵活的Notion格式转换：支持多轮消息、动态传参、context合并、ID策略合理。
    """
    now = datetime.now().isoformat()

    # context字段合并
    context_value = {
        "timezone": "Asia/Shanghai",
        "userName": "",
        "userId": user_id,
        "spaceName": "",
        "spaceId": space_id,
        "spaceViewId": space_view_id,
        "currentDatetime": now,
        "surface": "home_module"
    }
    if context_overrides:
        context_value.update(context_overrides)

    transcript = [
        {
            "id": "206a879c-fe5a-8010-b7db-00aa23a788a2",
            "type": "config",
            "value": {"type": "markdown-chat", "model": "openai-gpt-4.1"}
        },
        {
            "id": "206a879c-fe5a-80d8-b47e-00aa0b243212",
            "type": "context",
            "value": context_value
        },
        {
            "id": "206a879c-fe5a-8094-b892-00aab6d8a9e8",
            "type": "agent-integration"
        }
    ]
    
    transcript.append({
                "id": str(uuid.uuid4()),
                "type": "agent-integration"
    })

    # 支持多轮消息
    for msg in messages:
        if msg.role in ["user", "assistant"]:
            transcript.append({
                "id": str(uuid.uuid4()),
                "type": msg.role,
                "value": [[msg.content]],
                "userId": user_id,
                "createdAt": now
            })

    notion_request = {
        "traceId": trace_id,
        "spaceId": space_id,
        "transcript": transcript,
        "threadId": thread_id,
        "createThread": False,
        "debugOverrides": {"cachedInferences":{},"annotationInferences":{},"emitInferences":False},
        "generateTitle": True,
        "saveAllThreadOperations": True
    }
    print("DEBUG notion_request:", json.dumps(notion_request, ensure_ascii=False, indent=2))
    return notion_request

def parse_notion_response(line: str) -> Optional[Dict[str, Any]]:
    """解析Notion的NDJSON响应"""
    try:
        data = json.loads(line)
        if data.get("type") == "markdown-chat":
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": data.get("value", "")
                    },
                    "finish_reason": None
                }]
            }
        elif data.get("type") == "title":
            # 这是会话标题，可以忽略或作为元数据处理
            return None
    except json.JSONDecodeError:
        return None
    return None

async def stream_notion_to_openai(notion_request: Dict[str, Any]):
    """流式转换Notion响应为OpenAI格式"""
    print("开始流式转换Notion响应为OpenAI格式, 时间:", time.time())
    async with aiohttp.ClientSession() as session:
        async with session.post(
            NOTION_API_URL,
            headers=NOTION_HEADERS,
            json=notion_request
        ) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail="Notion API error")

            # 发送OpenAI流式响应开始
            yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'created': int(datetime.now().timestamp()), 'model': '', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\\n\\n"

            # 处理Notion的流式响应
            async for line in response.content:
                if line:
                    print("收到一行:", time.time(), line)
                    openai_chunk = parse_notion_response(line.decode('utf-8').strip())
                    if openai_chunk:
                        yield f"data: {json.dumps(openai_chunk)}\\n\\n"

            # 发送结束信号
            yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'created': int(datetime.now().timestamp()), 'model': '', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\\n\\n"
            yield "data: [DONE]\\n\\n"

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: None = Depends(verify_bearer_token)
):
    """OpenAI兼容的聊天完成接口"""

    # 转换请求格式
    notion_request = convert_to_notion_format(request.messages)

    if request.stream:
        # 流式响应
        return StreamingResponse(
            stream_notion_to_openai(notion_request),
            media_type="text/event-stream"
        )
    else:
        # 非流式响应
        async with aiohttp.ClientSession() as session:
            async with session.post(
                NOTION_API_URL,
                headers=NOTION_HEADERS,
                json=notion_request
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Notion API error")
                # 收集所有响应
                content = ""
                record_map_data = None  # 新增变量
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        print("DEBUG Notion NDJSON:", line_str)  # 调试输出
                        if not line_str:
                            continue  # 跳过空行
                        try:
                            data = json.loads(line_str)
                        except json.JSONDecodeError:
                            continue  # 跳过无法解析的行
                        if data.get("type") == "markdown-chat":
                            content += data.get("value", "")
                        elif data.get("type") == "record-map":
                            record_map_data = data  # 记录下来，后面备用
                # 如果 content 为空，尝试从 record-map 里提取
                if not content and record_map_data:
                    # 尝试提取 recordMap 里的最终回复
                    try:
                        record_map = record_map_data.get("recordMap", {})
                        thread_message = record_map.get("thread_message", {})
                        # 取第一个 message 的 step.value
                        for msg in thread_message.values():
                            value = msg.get("value", {})
                            step = value.get("step", {})
                            if step.get("type") == "markdown-chat":
                                content = step.get("value", "")
                                break
                    except Exception as e:
                        print("DEBUG record-map parse error:", e)

                # 如果依然没有内容，返回友好错误
                if not content:
                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion",
                        "created": int(datetime.now().timestamp()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "[未能从 Notion 响应中解析到内容，请检查请求参数或响应格式]"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                            "completion_tokens": 0,
                            "total_tokens": sum(len(msg.content.split()) for msg in request.messages)
                        }
                    }

                # 返回OpenAI格式的响应
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                        "completion_tokens": len(content.split()),
                        "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + len(content.split())
                    }
                }

@app.get("/v1/models")
async def list_models(
    _: None = Depends(verify_bearer_token)
):
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "openai-gpt-4.1",
                "object": "model",
                "created": 1687882410,
                "owned_by": "notion-proxy"
            },
            {
                "id": "anthropic-opus-4",
                "object": "model",
                "created": 1687882410,
                "owned_by": "notion-proxy"
            },
            {
                "id": "anthropic-sonnet-4",
                "object": "model",
                "created": 1687882410,
                "owned_by": "notion-proxy"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

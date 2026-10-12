import os
import time
import uuid
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Chat Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client_kwargs = {}
if "mongodb+srv://" in MONGO_URI or "ssl=true" in MONGO_URI.lower():
    client_kwargs["tlsCAFile"] = certifi.where()

client = AsyncIOMotorClient(MONGO_URI, **client_kwargs)
db = client["ai_chat_assistant"]
threads_collection = db["threads"]
messages_collection = db["messages"]

# OpenAI Client
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

TONE_PROMPTS = {
    "Professional": "You are a professional, polite, and formal executive assistant. Provide clear, well-structured, and articulate answers.",
    "Casual": "You are a friendly, conversational companion. Use warm, casual, everyday language and keep it approachable.",
    "Concise": "You are a direct, hyper-concise assistant. Provide the shortest possible complete answer. Do not use pleasantries or filler words."
}

class ChatRequest(BaseModel):
    thread_id: str | None = None
    prompt: str
    tone: str = "Professional"

@app.get("/api/threads")
async def get_threads():
    cursor = threads_collection.find({}, {"_id": 0}).sort("created_at", -1)
    threads = await cursor.to_list(length=100)
    return {"threads": threads}

@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Extra Feature 1: Thread cleanup/management."""
    await threads_collection.delete_one({"thread_id": thread_id})
    await messages_collection.delete_many({"thread_id": thread_id})
    return {"status": "success", "message": f"Thread {thread_id} deleted"}

@app.get("/api/threads/{thread_id}/messages")
async def get_messages(thread_id: str):
    cursor = messages_collection.find({"thread_id": thread_id}, {"_id": 0}).sort("timestamp", 1)
    messages = await cursor.to_list(length=200)
    return {"messages": messages}

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    system_instruction = TONE_PROMPTS.get(req.tone, TONE_PROMPTS["Professional"])
    now_iso = datetime.now(timezone.utc).isoformat()

    # Ensure thread exists
    existing = await threads_collection.find_one({"thread_id": thread_id})
    if not existing:
        title = req.prompt[:30].strip() + ("..." if len(req.prompt) > 30 else "")
        await threads_collection.insert_one({
            "thread_id": thread_id,
            "title": title,
            "created_at": now_iso
        })

    # Save user message
    user_msg_id = str(uuid.uuid4())
    await messages_collection.insert_one({
        "message_id": user_msg_id,
        "thread_id": thread_id,
        "role": "user",
        "content": req.prompt,
        "tone": req.tone,
        "timestamp": now_iso
    })

    # Fetch last 10 messages (sorted descending, then reversed for chronological context)
    cursor = messages_collection.find({"thread_id": thread_id}).sort("timestamp", -1)
    past_msgs = await cursor.to_list(length=10)
    past_msgs.reverse()

    api_messages = [{"role": "system", "content": system_instruction}]
    for msg in past_msgs:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    async def event_generator():
        start_time = time.time()
        full_ai_response = ""

        yield f"data: {json.dumps({'type': 'init', 'thread_id': thread_id})}\n\n"

        try:
            stream = await openai_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=api_messages,
                stream=True
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_ai_response += content
                    yield f"data: {json.dumps({'type': 'chunk', 'text': content})}\n\n"

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Persist assistant response
            await messages_collection.insert_one({
                "message_id": str(uuid.uuid4()),
                "thread_id": thread_id,
                "role": "assistant",
                "content": full_ai_response,
                "tone": req.tone,
                "latency_ms": elapsed_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            yield f"data: {json.dumps({'type': 'done', 'latency_ms': elapsed_ms})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

from agent_setup import run_agent_sync

app = FastAPI(title="Agent API")

# Allow requests from your Next.js dev server
origins = [
    "http://localhost:3000",
    "https://agent-with-nextjs.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    prompt: str

class AskResponse(BaseModel):
    reply: str

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    # Run the agent in thread executor to avoid blocking event loop (if it's CPU / blocking).
    # If run_agent_sync is lightweight, you may call directly.
    loop = asyncio.get_running_loop()
    try:
        # run_agent_sync is synchronous, so run in default executor
        reply = await loop.run_in_executor(None, run_agent_sync, prompt)
    except Exception as e:
        logging.exception("Agent failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return {"reply": reply or ""}

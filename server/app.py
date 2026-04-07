"""
FastAPI server exposing the OpenEnv HTTP interface for ExecutiveAssist-Env.

Endpoints:
  POST /reset          → reset environment, returns initial state
  POST /step           → submit an action, returns (state, reward, done, info)
  GET  /state          → get current state
  GET  /tasks          → list all available task IDs
  GET  /health         → liveness check
"""

import os
import json
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from my_env import ExecutiveAssistEnv

# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("executiveassist")

app = FastAPI(
    title="ExecutiveAssist-Env",
    description="An OpenEnv environment simulating an AI Executive Assistant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (stateful per-process; fine for evaluation)
_env: Optional[ExecutiveAssistEnv] = None


def get_env() -> ExecutiveAssistEnv:
    global _env
    if _env is None:
        _env = ExecutiveAssistEnv()
    return _env


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "ExecutiveAssist-Env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    env = get_env()
    return {
        "tasks": {
            "easy":   env.EASY_TASKS,
            "medium": env.MEDIUM_TASKS,
            "hard":   env.HARD_TASKS,
        },
        "all": env.ALL_TASKS,
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment. Optionally specify task_id and seed."""
    global _env
    _env = ExecutiveAssistEnv(
        task_id=req.task_id if req else None,
        seed=req.seed if req else 42,
    )
    try:
        state = _env.reset(
            task_id=req.task_id if req else None,
            seed=req.seed if req else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"reset() → task_id={state['task_id']}")
    return state


@app.post("/step")
def step(req: StepRequest):
    """Submit one action to the environment."""
    env = get_env()
    if env._current_task is None:
        raise HTTPException(status_code=400, detail="No task loaded. Call /reset first.")
    try:
        state, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"step({req.action.get('type')}) → reward={reward:.2f}, done={done}")
    return {
        "state":  state,
        "reward": reward,
        "done":   done,
        "info":   info,
    }


@app.get("/state")
def get_state():
    """Return current state without advancing the environment."""
    env = get_env()
    if env._current_task is None:
        raise HTTPException(status_code=400, detail="No task loaded. Call /reset first.")
    return env.state()


@app.get("/render")
def render():
    """Return a human-readable rendering of the environment."""
    env = get_env()
    return {"render": env.render()}


# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
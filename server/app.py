"""
FastAPI server exposing the OpenEnv HTTP interface for ExecutiveAssist-Env.
"""

import os
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

# Global env instance
_env: Optional[ExecutiveAssistEnv] = None


def get_env() -> ExecutiveAssistEnv:
    global _env
    if _env is None:
        _env = ExecutiveAssistEnv()
    return _env


# ──────────────────────────────────────────────
# Request Models
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
            "easy": env.EASY_TASKS,
            "medium": env.MEDIUM_TASKS,
            "hard": env.HARD_TASKS,
        },
        "all": env.ALL_TASKS,
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    global _env
    task_id = req.task_id if req else None
    seed = req.seed if req else 42

    # Validate task_id
    if task_id is not None:
        from tasks import TASKS
        if task_id not in TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id '{task_id}'. Valid options: {sorted(TASKS.keys())}"
            )

    _env = ExecutiveAssistEnv(task_id=task_id, seed=seed)

    try:
        state = _env.reset(task_id=task_id, seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"reset() → task_id={state['task_id']}")
    return state


@app.post("/step")
def step(req: StepRequest):
    env = get_env()

    if env._current_task is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    action = req.action

    if not isinstance(action, dict):
        raise HTTPException(status_code=422, detail="'action' must be a JSON object.")
    if "type" not in action:
        raise HTTPException(status_code=422, detail="Missing 'type' in action.")

    # sanitize inputs
    for key in action:
        if action[key] is not None:
            action[key] = str(action[key]).strip()

    try:
        state, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in step()")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"step({action.get('type')}) → reward={reward:.2f}, done={done}")

    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def get_state():
    env = get_env()
    if env._current_task is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.state()


@app.get("/render")
def render():
    env = get_env()
    return {"render": env.render()}


# ──────────────────────────────────────────────
# ENTRY POINT (IMPORTANT FOR BOTH HF + VALIDATOR)
# ──────────────────────────────────────────────

import uvicorn

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
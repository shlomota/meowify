#!/bin/bash
# FastAPI server — single worker required (JOBS dict is in-process memory)
cd ~/meowify-v2
source .venv/bin/activate
exec uvicorn server:app --host 127.0.0.1 --port 8504 --workers 1

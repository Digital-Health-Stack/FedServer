#!/bin/bash
concurrently -n BACKEND,FRONTEND -c yellow,cyan \
"cd backend/app && . venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload" \
"cd frontend/app && npm run dev"
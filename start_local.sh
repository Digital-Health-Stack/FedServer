#!/bin/bash
redis-server backend/utilities/redis.conf &

concurrently -n BACKEND,FRONTEND -c yellow,cyan \
"cd backend && . venv/bin/activate && PYTHONUNBUFFERED=1 uvicorn main:app --host 0.0.0.0 --port 8000 --reload" \
"cd frontend && npm run dev"
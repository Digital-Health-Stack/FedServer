#!/bin/bash

# Ensure below points:
# 1. You have a database.db with all tables in backend/app/storage
# 2. You have update .env in parent folder and frontend/app both
# 3. You envoke script with wslenv in ~ folder with all packages in backend/app/requirement.txt installed
# 4. npm install in /frontend/app

echo "Starting FastAPI backend at 0.0.0.0:8000..."
cd backend/app
setsid uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting React frontend..."
cd ../../frontend/app
setsid npm run dev &
FRONTEND_PID=$!

trap "echo 'Stopping...'; kill -TERM -$BACKEND_PID -$FRONTEND_PID" EXIT

wait
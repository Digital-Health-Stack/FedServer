#!/usr/bin/env bash
black backend/app
prettier --write "frontend/app/**/*.{js,ts,jsx,tsx,json,css,html,md}"
echo "🍻 All code formatted!"

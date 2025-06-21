#!/usr/bin/env bash

# Create folder .vscode if it doesn’t exist.
# Inside it, create or open settings.json.

# Paste this inside:
# {
#   "editor.formatOnSave": true,

#   "python.formatting.provider": "black",
#   "python.formatting.blackArgs": ["--line-length", "88"],

#   "editor.defaultFormatter": "esbenp.prettier-vscode",

#   "[python]": {
#     "editor.formatOnSave": true
#   },
#   "[javascript]": {
#     "editor.formatOnSave": true
#   },
#   "[typescript]": {
#     "editor.formatOnSave": true
#   },
#   "[json]": {
#     "editor.formatOnSave": true
#   },
#   "[html]": {
#     "editor.formatOnSave": true
#   },
#   "[css]": {
#     "editor.formatOnSave": true
#   }
# }

black backend/app
prettier --write "frontend/app/**/*.{js,ts,jsx,tsx,json,css,html,md}"
echo "🍻 All code formatted!"

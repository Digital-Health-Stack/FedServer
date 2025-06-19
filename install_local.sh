#!/bin/bash

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print banners
print_banner() {
  COLOR=$1
  TEXT=$2
  echo -e "${COLOR}"
  echo "============================================"
  echo "        $TEXT"
  echo "============================================"
  echo -e "${NC}"
}

# --- Backend Setup ---
print_banner "$BLUE" "Installing FastAPI backend..."
cd backend/app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p storage
alembic upgrade head
print_banner "$GREEN" "Backend installed"

# --- Frontend Setup ---
cd ../../frontend/app
print_banner "$BLUE" "Installing React frontend..."
npm install
print_banner "$GREEN" "Frontend installed"

# Back to root
cd ../..

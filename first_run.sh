#!/bin/bash

# --- CONFIGURATION ---
VENV_NAME="venv"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   🧬 Word Algebra Setup Wizard   ${NC}"
echo -e "${BLUE}=======================================${NC}"

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${GREEN}[1/4] Creating virtual environment ($VENV_NAME)...${NC}"
    python3 -m venv $VENV_NAME
else
    echo -e "${GREEN}[1/4] Virtual environment exists. Skipping creation.${NC}"
fi

# 3. Activate Environment
# This syntax works for both Bash and ZSH
echo -e "${GREEN}[2/4] Activating environment...${NC}"
source $VENV_NAME/bin/activate

# 4. Install Dependencies
echo -e "${GREEN}[3/4] Installing dependencies...${NC}"
pip install --upgrade pip > /dev/null
pip install -r requirements.txt

# 5. Run Data Generation (setup.py)
echo -e "${GREEN}[4/4] Running setup.py (This downloads models & builds the matrix)...${NC}"
echo -e "${BLUE}Note: The first run can take 10-20 minutes to download the 400k dictionary.${NC}"
python setup.py

# 6. Success Message
echo -e "${BLUE}=======================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "You can now run the server with:"
echo -e "${BLUE}source venv/bin/activate && python app.py${NC}"
echo -e "${BLUE}=======================================${NC}"

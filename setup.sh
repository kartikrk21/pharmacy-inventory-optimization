echo "================================================"
echo "Pharmacy Inventory Optimization System Setup"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python $python_version detected${NC}"
else
    echo -e "${RED}✗ Python 3.8+ required${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p ml_models/trained_models
mkdir -p data_generation/data
mkdir -p logs
mkdir -p webapp/static/downloads
mkdir -p webapp/static/uploads
mkdir -p database/backups
echo -e "${GREEN}✓ Directories created${NC}"

# Set environment variables
echo -e "${YELLOW}Setting environment variables...${NC}"
export DATABASE_URL=${DATABASE_URL:-"sqlite:///$(pwd)/pharmacy.db"}
echo "DATABASE_URL=$DATABASE_URL" > .env
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> .env
echo -e "${GREEN}✓ Environment variables set${NC}"

# Start Docker services
echo -e "${YELLOW}Starting Docker services...${NC}"
docker-compose up -d
sleep 10
echo -e "${GREEN}✓ Docker services started${NC}"

# Initialize database
echo -e "${YELLOW}Initializing database...${NC}"
python3 - <<PY
from config.config import Config
from database.db_manager import DatabaseManager

db = DatabaseManager(Config.DATABASE_URL)
db.init_db()
print('✓ Database initialized')
PY

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. python webapp/app.py"
echo "3. Open http://localhost:5000"
echo ""
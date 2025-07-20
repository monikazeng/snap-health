# snap-health

## ⚙️ Getting Started

## Train new data model
run command:
 python backend_ml/train.py

## Running the Backend API
### 1. Clone the repository
cd your-repo-name/backend

### 2. Create and activate a virtual environment
- in macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the FastAPI server
uvicorn main:app --reload
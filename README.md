# 🚀 RAG Chatbot

## 🛠️ Installation & Setup

### 1. Clone the Repository & Install Dependencies
Clone the repository
```bash
git clone https://github.com/AhmadHammad21/AI-Agent.git
cd AI-Agent
```

### 2. Create and activate a virtual environment
You can use anaconda environment as well 
### Windows
```bash
python -m venv venv
venv/Scripts/activate
```

### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```


### 4. (Optional) Enable LangSmith for Model Monitoring

To enable LangSmith tracing, create a .env file and add your API key and project name:
```bash
OPENAI_API_KEY=""
```

# 🚀 Running the Chatbot

### Run the Streamlit Chatbot UI
```bash
streamlit run app.py
```


### Run the FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```


### Run via Docker
```bash
# Build the Docker image
docker build -t rag-api .

# Run the container
docker run -p 8000:8000 rag-api
```


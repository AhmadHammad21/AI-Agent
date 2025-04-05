# 🤖 Chatbot

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


### 3. Add OpenAI API Key 

Create a .env file and add your API key and project name:
```bash
OPENAI_API_KEY=""
MONGODB_URL="mongodb://34.173.119.32/"
MONGODB_DATABASE="ai-agent"
MONGODB_COLLECTION="chatbot"
```


## 🚀 Running the Chatbot

### Run the Docker Compose
```bash
sudo docker-compose up --build
```

### Run the FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

### Run the Streamlit Chatbot UI
```bash
streamlit run app.py
```





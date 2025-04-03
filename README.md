# 🚀 RAG Chatbot
**100% Open Source | Local PC Installation**

## 🔥 RAG: The Ultimate RAG Stack!

The * RAG Chatbot** is a powerful tool designed for fast, accurate, and explainable retrieval of information from PDFs. Leveraging *-7B**, **FAISS**, and **Chat History Integration**, this chatbot provides a seamless experience for document-based question answering and information retrieval.

## 🛠️ Tech Stack

- **Core Model**:-7B
- **Embedding Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Framework**: Python
- **Model Deployment**: Ollama
- **Retrieval & Generation**: LangChain
- **Monitoring & Debugging**: LangSmith
- **UI Framework**: Streamlit
- **API Framework**: FastAPI
- **Vector Database**: FAISS
- **Containerization**: Docker
---

## 🛠️ Installation & Setup

### 1. Clone the Repository & Install Dependencies
Clone the repository
```bash
git clone https://github.com/AhmadHammad21-RAG-ChatBot
cd-RAG-ChatBot
```

### 2. Create and activate a virtual environment
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
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="YOUR_API_KEY"
LANGSMITH_PROJECT="YOUR_PROJECT_NAME"
```

# 🚀 Running the Chatbot

### Run the Streamlit Chatbot UI
```bash
streamlit run app.py
```
🖼️ Example:
![Streamlit UI Screenshot](images/streamlit.png)


### Run the FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```
📌 Access Swagger API Documentation: http://localhost:5000/docs
🖼️ Example:
![Swagger UI Screenshot](images/swagger.png)


### Run via Docker
```bash
# Build the Docker image
docker build -t rag-api .

# Run the container
docker run -p 8000:8000 rag-api
```

## How It Works

- Upload Documents: Place your PDFs in the `data/docs/ ` directory.

- Run the Application: Start the Streamlit UI or FastAPI server.

- Retrieval & Generation: The chatbot retrieves the most relevant document chunks and generates responses using -7B`.

### Future Enhancements

- 📂 Support additional document formats (DOCX, TXT, etc.).

- 🔍 Implement Neural Reranking for improved search accuracy.

- 🤖 Explore HyDE (Hypothetical Document Embeddings).

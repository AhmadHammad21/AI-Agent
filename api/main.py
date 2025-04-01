from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.rag_pipeline import RAGPipeline
from src.embeddings.vector_store import VectorStore
from config.settings import VECTOR_STORE_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME

app = FastAPI()

vector_store = VectorStore(EMBEDDING_MODEL_NAME)

vector_store.load_vector_store(VECTOR_STORE_PATH)

rag_pipeline = RAGPipeline(vector_store, model_name=LLM_MODEL_NAME)

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def get_answer(request: QueryRequest):
    response = rag_pipeline.retrieve_and_generate(request.query)
    return {"response": response}

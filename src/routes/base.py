from fastapi import FastAPI, APIRouter, Depends
from config.config import get_settings, Settings
from config.settings import settings
from fastapi import FastAPI
from pydantic import BaseModel
from retrieval.rag_pipeline import RAGPipeline
from embeddings.vector_store import VectorStore

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)


class QueryRequest(BaseModel):
    query: str

@base_router.post("/query/")
async def get_answer(request: QueryRequest):
    vector_store = VectorStore(settings.EMBEDDING_MODEL_NAME)

    vector_store.load_vector_store(settings.VECTOR_STORE_PATH)

    rag_pipeline = RAGPipeline(vector_store, model_name=settings.LLM_MODEL_NAME)
    response = rag_pipeline.retrieve_and_generate(request.query)
    return {"response": response}


@base_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):

    app_name = settings.EMBEDDING_MODEL_NAME
    # app_version = app_settings.APP_VERSION

    return {
        "app_name": app_name
    }
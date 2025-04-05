from fastapi import FastAPI, APIRouter, status, Request
from config.config import config
from config.settings import settings
from fastapi import FastAPI
from pydantic import BaseModel
# from retrieval.rag_pipeline import RAGPipeline
# from embeddings.vector_store import VectorStore
from fastapi.responses import JSONResponse

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)


@base_router.get("/")
async def welcome():

    return {
        "status": "Healthy"
    }
from fastapi import FastAPI
from routes import base
from config.config import config
from config.settings import settings
from llms.llm_provider_factory import LLMProviderFactory

async def lifespan(app: FastAPI):

    llm_provider_factory = LLMProviderFactory(config=config, settings=settings)

    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)

    yield  # This is where FastAPI runs the application

    # Closing connections

app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
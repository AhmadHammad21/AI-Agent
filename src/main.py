from fastapi import FastAPI
from routes import base
from config.config import get_settings

async def lifespan(app: FastAPI):
    settings = get_settings()
    
    yield  # This is where FastAPI runs the application

app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
from fastapi import FastAPI, APIRouter, status, Request
from config.config import config
from config.settings import settings
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.response_signal  import ResponseSignal


nlp_router = APIRouter(
    prefix="/api/v1/chatbot",
    tags=["api_v1"]
)

class AnswerRequest(BaseModel):
    user_id: str
    session_id: str
    query: str

@nlp_router.post("/answer")
async def answer_rag(request: Request, answer_request: AnswerRequest):

    user_id = answer_request.user_id
    session_id = answer_request.session_id
    query = answer_request.query

    # Call the RAG client with user_id, session_id, and query # TODO: ADD USER-ID, REPORT-ID TO
    # TO RETRIEVE HISTORY
    answer, full_prompt, chat_history = request.app.rag_client.answer_rag_question(
        query=query
    )

    # if not answer:
    #     return JSONResponse(
    #             status_code=status.HTTP_400_BAD_REQUEST,
    #             content={
    #                 "signal": ResponseSignal.RAG_ANSWER_ERROR.value
    #             }
    #     )
    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,  
            "query": query,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )

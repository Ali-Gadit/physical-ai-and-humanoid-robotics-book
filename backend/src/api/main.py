from fastapi import FastAPI, Request, Response, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from chatkit.server import StreamingResult
import asyncio

from src.chatkit_integration import chatkit_server
from src.middleware.auth import get_current_user

app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware to allow requests from the frontend (Docusaurus)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Restrict to frontend for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@app.post("/chatkit")
async def chatkit_endpoint(request: Request, user: dict = Depends(get_current_user)):
    """
    ChatKit endpoint that handles all chat interactions using the ChatKit Protocol.
    Protected by Better Auth session.
    """
    # Process the request using the ChatKit server
    # context can be used to pass session info, auth headers, etc.
    context = {"user": user} 
    result = await chatkit_server.process(await request.body(), context=context)
    
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    else:
        return Response(content=result.json, media_type="application/json")
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from chatkit.server import StreamingResult
import asyncio

from src.chatkit_integration import chatkit_server

app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware to allow requests from the frontend (Docusaurus)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, specify domains.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    """
    ChatKit endpoint that handles all chat interactions using the ChatKit Protocol.
    """
    # Process the request using the ChatKit server
    # context can be used to pass session info, auth headers, etc.
    context = {} 
    result = await chatkit_server.process(await request.body(), context=context)
    
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    else:
        return Response(content=result.json, media_type="application/json")
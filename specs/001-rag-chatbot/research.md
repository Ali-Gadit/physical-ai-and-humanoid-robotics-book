# Research Notes: OpenAI Agents SDK & ChatKit with Gemini

**Date**: 2025-12-07
**Context**: Investigating how to use OpenAI Agents SDK and ChatKit with Gemini models for the RAG Chatbot feature.

## OpenAI Agents SDK (Python)

**Library ID**: `/openai/openai-agents-python`

### Core Concept
The SDK uses `Agent` objects that are run by a `Runner`. Agents can have `tools` and `handoffs`.

### Gemini Integration (via LiteLLM)
The SDK has built-in support for `LiteLLM`, which allows using non-OpenAI models like Gemini.

**Installation**:
```bash
pip install "openai-agents[litellm]"
# OR
pip install litellm
```

**Usage Code Pattern**:
```python
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

# Configure Agent to use Gemini
agent = Agent(
    name="Gemini Assistant",
    instructions="You are a helpful assistant for the Robotics textbook.",
    model=LitellmModel(
        model="gemini/gemini-1.5-flash",  # Use appropriate Gemini model version
        api_key="<GEMINI_API_KEY>"        # From env vars
    )
)

# Running the agent
result = await Runner.run(agent, "What is Physical AI?")
```

### RAG Integration
The SDK supports `FileSearchTool`, but for our specific Qdrant + Gemini Embedding setup, we should likely create a **custom tool** or use the `Agent`'s context injection capabilities if we are doing the retrieval manually before calling the agent.

Given the requirements, a custom function tool is best:
```python
@function_tool
async def search_book_content(query: str):
    """Searches the textbook for relevant info."""
    # Logic to embed query with Gemini and search Qdrant
    return retrieved_chunks
```

## OpenAI ChatKit (Frontend - React)

**Library ID**: `/openai/chatkit-js` (and `@openai/chatkit-react`)

### Core Concept
Provides a pre-built UI for chat interfaces (`<ChatKit />`) and hooks (`useChatKit`) to connect to a backend.

### Setup
1.  **Install**: `npm install @openai/chatkit-react`
2.  **Configuration**:
    ```javascript
    import { ChatKit, useChatKit } from '@openai/chatkit-react';

    function Chatbot() {
      const { control } = useChatKit({
        api: {
          // Endpoint that returns a client token or handles the session
          getClientSecret: async () => {
             const res = await fetch('/api/chat/session');
             return (await res.json()).client_secret;
          }
        },
        // Customization
        startScreen: { greeting: "Ask about the book..." }
      });

      return <ChatKit control={control} />;
    }
    ```

## OpenAI ChatKit (Backend - Python)

**Library ID**: `/openai/chatkit-python`

### Role
The Python SDK for ChatKit (`chatkit-python`) seems to provide a server implementation (`ChatKitServer`) that bridges the frontend request to the Agents SDK `Runner`.

**Key Flow**:
1.  Frontend `useChatKit` connects to Backend.
2.  Backend `ChatKitServer` receives message.
3.  Backend uses `Agents SDK` (with Gemini) to generate response.
4.  Backend streams response back to Frontend.

*Note*: If `chatkit-python` is complex to set up with custom LiteLLM agents, we might fallback to a manual FastAPI endpoint that streams responses compatible with what ChatKit expects, or just use `Runner.run_streamed` and format the output.

## Recommended Architecture for Implementation

1.  **Backend**:
    *   **FastAPI** app.
    *   **Agent**: Configured with `LitellmModel` (Gemini).
    *   **Tools**: Custom `search_book` tool (wrapping Qdrant retrieval).
    *   **Endpoint**: `/chat` (websocket or streaming HTTP) that uses `Runner` to execute the agent and streams back results.

2.  **Frontend**:
    *   **React** app.
    *   **ChatKit**: Configured to talk to the FastAPI endpoint.

## Action Items
1.  Add `litellm` to `backend/requirements.txt`.
2.  Implement `backend/src/services/chat_service.py` using `LitellmModel`.
3.  Implement Custom Tool for RAG in `backend/src/rag/tools.py`.

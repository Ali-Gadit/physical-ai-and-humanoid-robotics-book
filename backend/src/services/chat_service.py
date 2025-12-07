import os
from agents import Agent, Runner, function_tool , set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from src.rag.retrieval import retrieve_context

set_tracing_disabled(True)  # Disable tracing for performance
load_dotenv()

# Define the tool for the agent
@function_tool
def search_textbook(query: str, focus_text: str = None) -> str:
    """
    Search the 'Physical AI and Humanoid Robotics' textbook for information.
    Use this tool whenever the user asks a question that might be answered by the book's content.
    If `focus_text` is provided, prioritize searching within or very close to that specific text.
    The output is a string containing relevant text chunks from the book.
    """
    print(f"[DEBUG] Agent searching textbook for: {query}, with focus on: {focus_text}")
    # Augment the query to prioritize the focus_text
    augmented_query = f"{query} based on the following specific text: {focus_text}" if focus_text else query
    return retrieve_context(augmented_query) # retrieve_context already handles the actual search


def get_litellm_model() -> LitellmModel:
    """
    Returns the configured LiteLLM model for the agent.
    Updated to use gemini/gemini-2.5-flash-lite.
    """
    return LitellmModel(
        model="gemini/gemini-2.5-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY")
    )

def get_agent_instructions() -> str:
    """
    Returns the instructions for the RAG agent.
    """
    return (
        "You are an expert AI tutor for the 'Physical AI and Humanoid Robotics' textbook. "
        "Your goal is to answer student questions accurately based ONLY on the provided textbook content. "
        "1. ALWAYS use the `search_textbook` tool when user asks about a specific topic about Physical AI and Humanoid Robotics. "
        "2. If the user provides a 'quoted_text' (indicated by 'The user is referring to this in particular: [text]'), "
        "   you MUST use that text as the `focus_text` argument when calling the `search_textbook` tool. "
        "   For example, if the user says 'What is this?' and you see 'The user is referring to this in particular: Some selected text', "
        "   you should call `search_textbook(query='What is this?', focus_text='Some selected text')`. "
        "3. If the search results contain the answer, summarize it clearly and cite the source if available. "
        "4. If the search results do NOT contain the answer, politely state that the information is not found in the textbook. "
        "5. Do not hallucinate or make up information outside the textbook context. "
        "6. Be encouraging and concise."
    )

def get_rag_agent() -> Agent:
    """
    Returns the configured RAG agent.
    """
    return Agent(
        name="RoboticsTutor",
        instructions=get_agent_instructions(),
        model=get_litellm_model(),
        tools=[search_textbook]
    )

# Instantiate the agent globally or pass it around
agent = get_rag_agent()

async def get_chat_response(user_message: str) -> str:
    """
    Process a user message through the agent and return the response.
    This function is now less critical as ChatKitServer will handle agent interaction.
    """
    try:
        result = await Runner.run(agent, user_message)
        return result.final_output
    except Exception as e:
        print(f"Error in chat service: {e}")
        return "I encountered an error while processing your request."
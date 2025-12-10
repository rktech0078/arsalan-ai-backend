# agent_setup.py
import os
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, Runner

load_dotenv()

# Load GROQ / model credentials from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")

# Create client and model (same as your main.py)
openrouter_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

openrouter_model = OpenAIChatCompletionsModel(
    model="openrouter/auto",
    openai_client=openrouter_client 
)

# Create the agent
openrouter_agent = Agent(
    name="Professional Assistant",
    instructions="You are a helpful, professional, and intelligent AI assistant. Your goal is to provide clear, accurate, and concise answers to the user.",
    model=openrouter_model,
)


# openrouter/auto

set_tracing_disabled(True)

# Helper to run agent synchronously (wrap Runner.run_sync)
def run_agent_sync(prompt: str, timeout_seconds: int | None = None):
    """
    Run the agent synchronously and return final_output string.
    If your Runner.run_sync supports additional args (timeout, max tokens), add them here.
    """
    # You might change this to an async call if your Runner supports it.
    result = Runner.run_sync(
        starting_agent=openrouter_agent,
        input=prompt
    )
    # result may have different attributes depending on your agents package; adjust if needed
    return getattr(result, "final_output", str(result))

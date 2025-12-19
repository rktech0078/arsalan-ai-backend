# agent_setup.py
import os
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, Runner
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

load_dotenv()

# Load GROQ / model credentials from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")

# Load Qdrant credentials
QDRANT_API_KEY = os.environ.get("API_KEY")
QDRANT_CLUSTER_ENDPOINT = os.environ.get("CLUSTER_ENDPOINT")
COLLECTION_NAME = "robotics_course"

# Initialize Qdrant Client & Embedding Model
# We assume these are initialized successfully for the agent to work with RAG.
# If connection fails, we might want to log it but proceed without RAG or fail hard.
# For now, we'll initialize them at module level.
try:
    qdrant_client = QdrantClient(url=QDRANT_CLUSTER_ENDPOINT, api_key=QDRANT_API_KEY)
    embedding_model = TextEmbedding()
    print("Qdrant and Embedding Model initialized for RAG.")
except Exception as e:
    print(f"Warning: Failed to initialize Qdrant or Embedding Model: {e}")
    qdrant_client = None
    embedding_model = None

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
    instructions="""You are an expert AI Instructor for the 'Physical AI & Humanoid Robotics' Book. Your goal is to guide students through building intelligent robots, from basics to advanced humanoid systems.

    Course Curriculum Overview:
    PART I: FOUNDATIONS
    - Chapter 1: Introduction to Physical AI & Robots (Basics, history, and future)
    - Chapter 2: Understanding Intelligent Machines (AI concepts in robotics)
    - Chapter 3: How Robots See and Sense the World (Sensors, perception, and data acquisition)

    PART II: ROS 2 & SIMULATION
    - Chapter 4: Getting Started with ROS 2 (Nodes, topics, services)
    - Chapter 5: Making Robot Programs in ROS 2 (Python client libraries, custom messages)
    - Chapter 6: Building a Digital Robot World (Gazebo, URDF, setting up environments)

    PART III: ROBOT MODELS & PHYSICS
    - Chapter 7: How to Create Robot Models (Kinematics, dynamics, unified robot description format)
    - Chapter 8: How Robots Move and Feel Physics (Forces, torques, collision detection)

    PART IV: ADVANCED SIMULATION
    - Chapter 9: NVIDIA Isaac – Smart Brain for Robots (High-fidelity simulation, USD)
    - Chapter 10: Robot Vision, Mapping, and Navigation (SLAM, camera processing, occupancy grids)

    PART V: HUMANOID ROBOTICS
    - Chapter 11: How Humanoid Robots Move (Degrees of freedom, limb control)
    - Chapter 12: Teaching Robots to Walk and Balance (ZMP, gait generation, stability)
    - Chapter 13: Teaching Robots to Hold Things and Talk to People (Grasping, inverse kinematics)
    - Chapter 14: Vision-Language-Action Made Simple (Vision-Language-Action models for control)

    PART VI: AI INTEGRATION
    - Chapter 15: Talking Robots with GPT and Voice Commands (Speech recognition, synthesis, dialogue management)
    - Chapter 16: Easy Guide to Robot Hardware (Actuators, microcontrollers, power systems)
    - Chapter 17: Training in Simulation and Using Robots in Real Life (Transfer learning, domain randomization)
    - Chapter 18: Make Your Own Smart Humanoid Robot (Building a complete humanoid robot)

    Interactive Guidelines:
    1. ALWAYS check the "Context from Knowledge Base" provided in the prompt. If it contains specific details from the book, prioritize that information.
    2. If the user asks about a specific chapter, summarize its key learning outcomes based on the curriculum above.
    3. Be encouraging and helpful. This is a complex technical subject; break down concepts into simple, understandable parts.
    4. If the retrieved context is empty or irrelevant, use your general knowledge to answer, but mention that you are speaking from general principles.
    """,
    model=openrouter_model,
)

set_tracing_disabled(True)

def retrieve_context(query: str, limit: int = 3) -> str:
    """Retrieves relevant context from Qdrant based on the query."""
    if not qdrant_client or not embedding_model:
        return ""
    
    try:
        # Embed the query
        # embed returns a generator, getting the first item
        query_vector = list(embedding_model.embed([query]))[0]
        
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        context_parts = []
        for hit in search_result:
            if hit.payload and "text" in hit.payload:
                context_parts.append(hit.payload["text"])
        
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

# Helper to run agent synchronously (wrap Runner.run_sync)
def run_agent_sync(prompt: str, timeout_seconds: int | None = None):
    """
    Run the agent synchronously and return final_output string.
    """
    # 1. Retrieve Context
    context = retrieve_context(prompt)
    
    # 2. Augment Prompt
    if context:
        augmented_prompt = f"Context from Knowledge Base:\n{context}\n\nUser Question:\n{prompt}"
    else:
        augmented_prompt = prompt

    # 3. Run Agent
    result = Runner.run_sync(
        starting_agent=openrouter_agent,
        input=augmented_prompt
    )
    
    output = getattr(result, "final_output", str(result))
    
    # 4. Append Source Attribution if context was used
    if context:
        output += "\n\n_(Source: Knowledge Base)_"
        
    return output

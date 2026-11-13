import os
import json
import chromadb

from openai import OpenAI
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY"),
)

MODEL = "nvidia/nemotron-3.5-lightning:free"       # chat model (free, NVIDIA)
EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-8b:free"
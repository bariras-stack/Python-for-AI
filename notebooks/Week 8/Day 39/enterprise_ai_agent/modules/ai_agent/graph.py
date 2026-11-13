from modules.ai_agent.tools import Tools
from core.services.LLM import llm_client, MODEL, EMBEDDING_MODEL
from langgraph.graph import START, MessageState, StateGraph
from langchain_core.messages import Messages, SystemMessage, HumanMessage, AIMessage

SYSTEM_PROMPT = """You are the friendly customer support assistant for Xander Restaurant.
Keep replies short. Prices are in Pakistani rupees. Use fetch_menu for menu/prices.
For complaints, collect customer name, phone, complaint and order ID. Validate the
order ID with validate_order_id, then call log_complaint only for a valid order.
Never invent facts. Ask for missing details."""

def create_chat_model(llm_client=llm_client, model=MODEL):
    return llm_client.chat.models.create(model=model)

def assistant(state:MessageState, model_with_tools):
    reply = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)], *state["messages"])
    return {"messages": [reply]}

def build_graph(model=None, tools=None):
    tools = Tools if tools is None else tools
    model=create_chat_model()

model_with_tools = model.blind_tools(tools)



from langchain_core.agents import AgentExecutor, Tool
#def build_graph(model=None, tools=None):
    #tools=Tools if tools is None else tools


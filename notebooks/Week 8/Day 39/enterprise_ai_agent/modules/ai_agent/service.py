from langchain_core.messages import HumanMessage
import re

from core.database.sqlite import (
    load_messages,
    save_messages
)

from modules.ai_agent.graph import graph
from modules.ai_agent.offer_agent import offer_graph


OFFER_REQUEST_PATTERN = re.compile(
    r"\b(offers?|deals?|discounts?|promotions?|promos?|specials?)\b",
    re.IGNORECASE,
)


def process_chat(query: str, session_id: str):

    # 1. Load previous conversation
    previous_messages = load_messages(session_id)

    # 2. Add current user message
    user_message = HumanMessage(content=query)

    conversation = previous_messages + [user_message]

    # 3. Execute LangGraph
    if OFFER_REQUEST_PATTERN.search(query):
        offer_result = offer_graph.invoke({"messages": [user_message]})
        result = {"messages": conversation + [offer_result["messages"][-1]]}
    else:
        result = graph.invoke({
            "messages": conversation
        })

    # 4. Get newly generated messages
    new_messages = result["messages"][len(previous_messages):]

    # 5. Save user, assistant and tool messages
    save_messages(session_id, new_messages)

    # 6. Get final assistant response
    final_response = result["messages"][-1].content

    return {
        "session_id": session_id,
        "response": final_response
    }


def retrieve_history(session_id: str):

    messages = load_messages(session_id)

    history = []

    for message in messages:

        history.append({
            "role": message.type,
            "content": message.content
        })

    return {
        "session_id": session_id,
        "messages": history
    }

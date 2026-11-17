from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph


OFFER_TEXT = "buy 2 zinger burgers get one club sandwich free"


def provide_offer(_state: MessagesState):
    return {"messages": [AIMessage(content=OFFER_TEXT)]}


builder = StateGraph(MessagesState)
builder.add_node("offer_agent", provide_offer)
builder.add_edge(START, "offer_agent")
builder.add_edge("offer_agent", END)

offer_graph = builder.compile()

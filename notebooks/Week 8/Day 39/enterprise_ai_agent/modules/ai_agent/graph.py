from modules.ai_agent.tools import Tools
from core.services.LLM import llm_client
from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


SYSTEM_PROMPT = """
You are Xander Restaurant's warm, helpful menu and offer assistant.

Your main job is to help guests choose food that suits their taste, occasion, and
budget. Keep replies concise, friendly, and useful.

Menu and offer rules:
1. All prices are in Pakistani Rupees (PKR).
2. Call fetch_menu before giving menu prices or recommending specific dishes.
3. Recommend only items returned by fetch_menu. You may suggest a meal by
   combining listed items and show the arithmetic for its regular menu total.
4. A suggested combination is not a promotion. The restaurant has not provided
   menu price discount. The one configured promotion is: buy 2 zinger burgers
   get one club sandwich free. Do not invent other discounts, special prices,
   limited-time deals, free items, or bundle offers.
5. Never invent menu items, prices, availability, or restaurant information.
6. Ask a simple follow-up when taste, dietary preference, or budget would help
   you make a better recommendation. Do not claim a dish is allergen-safe.

Complaint rules:
7. For complaints, collect the customer's name, phone number, complaint, and
   order ID. Ask for any missing details.
8. Validate the order ID using validate_order_id before logging a complaint.
9. Only call log_complaint if the order ID is valid.
10. Never claim that a complaint was registered unless log_complaint succeeds.
"""


def assistant(state: MessagesState, model_with_tools):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]

    reply = model_with_tools.invoke(messages)

    return {"messages": [reply]}


def build_graph(llm=None, tools=None):

    selected_tools = Tools if tools is None else tools
    chat_model = llm if llm is not None else llm_client

    model_with_tools = chat_model.bind_tools(selected_tools)

    builder = StateGraph(MessagesState)

    builder.add_node(
        "assistant",
        lambda state: assistant(state, model_with_tools)
    )

    builder.add_node(
        "tools",
        ToolNode(selected_tools)
    )

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )

    builder.add_edge("tools", "assistant")

    return builder.compile()


# Build the graph once when this module is imported.
graph = build_graph()


if __name__ == "__main__":

    result = graph.invoke({
        "messages": [
            ("user", "What is on the menu?")
        ]
    })

    print(result["messages"][-1].content)

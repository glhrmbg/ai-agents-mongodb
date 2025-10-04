import os
from typing import Annotated, List
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import tool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
from typing_extensions import TypedDict


# ============================================
# CONFIGURATION
# ============================================

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = "ai_agents"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Validation
if not MONGODB_URI or not OPENAI_API_KEY:
    raise ValueError("MONGODB_URI and OPENAI_API_KEY environment variables are required!")


# ============================================
# DATABASE
# ============================================

# MongoDB connection
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
vs_collection = db["chunked_docs"]
full_collection = db["full_docs"]

# Embedding model
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=1536)


def generate_embeddings(text: str) -> List[float]:
    """Generates embeddings for a given text"""
    return embedding_model.embed_query(text)


def vector_search(query: str, limit: int = 5) -> str:
    """Performs vector search in MongoDB"""
    query_embedding = generate_embeddings(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_search",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "body": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = vs_collection.aggregate(pipeline)
    docs = [doc.get("body", "") for doc in results]

    return "\n\n".join(docs) if docs else "No relevant documents found."


def get_full_page(title: str) -> str:
    """Retrieves full page content by title"""
    doc = full_collection.find_one({"title": title}, {"_id": 0, "body": 1})

    if doc:
        return doc["body"]
    return f"Page '{title}' not found."


# ============================================
# TOOLS
# ============================================

@tool
def search_information(question: str) -> str:
    """
    Searches for relevant information in the documentation to answer questions.
    Use this tool for general questions.

    Args:
        question: The user's question
    """
    return vector_search(question)


@tool
def get_page_summary(title: str) -> str:
    """
    Retrieves the complete content of a specific page for summarization.
    Use when the user requests a summary of a specific page.

    Args:
        title: The exact title of the page
    """
    return get_full_page(title)


TOOLS = [search_information, get_page_summary]


# ============================================
# AGENT (LANGGRAPH)
# ============================================

# Graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]


# LLM configuration with tools
llm = ChatOpenAI(temperature=0, model=LLM_MODEL)

prompt = ChatPromptTemplate.from_messages([
    (
        "You are a helpful AI assistant."
        "You are provided with tools to answer questions and summarize technical documentation related to data.\n"
        "Think step-by-step and use these tools to get the information required to answer the user query.\n"
        "Do not re-run tools unless absolutely necessary.\n"
        "If you are not able to get enough information using the tools, reply with I DON'T KNOW.\n"
        "You have access to the following tools: {tool_names}."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

tool_names = ", ".join([t.name for t in TOOLS])
prompt = prompt.partial(tool_names=tool_names)
llm_with_tools = prompt | llm.bind_tools(TOOLS)

tools_by_name = {tool.name: tool for tool in TOOLS}


# Graph nodes
def agent_node(state: State):
    """Processes the message and decides whether to use tools"""
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}


def tool_node(state: State):
    """Executes the requested tools"""
    results = []
    tool_calls = state["messages"][-1].tool_calls

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    return {"messages": results}


def should_continue(state: State):
    """Decides whether to continue to tools or finish"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_graph():
    """Creates the agent graph with persistent memory"""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("agent", should_continue)

    # Checkpoint for memory
    checkpointer = MongoDBSaver(client)
    return graph.compile(checkpointer=checkpointer)


# ============================================
# INTERFACE
# ============================================

def chat(app, message: str, thread_id: str = "1"):
    """Sends a message to the agent and returns the response"""
    input_msg = {"messages": [("user", message)]}
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"YOU: {message}")
    print(f"{'='*60}")

    # Process
    for output in app.stream(input_msg, config):
        for key in output.keys():
            print(f"🔄 Executing: {key}")

    # Final response
    response = output[key]["messages"][-1].content
    print(f"\n🤖 ASSISTANT:\n{response}\n")

    return response


# ============================================
# EXECUTION
# ============================================

def main():
    """Program entry point"""
    print("🚀 Starting RAG Agent with MongoDB...\n")

    # Build the graph
    app = build_graph()

    # Usage examples
    chat(app, "Give me a summary of the page 'Create a MongoDB Deployment'", thread_id="session_1")
    chat(app, "What did I just ask you?", thread_id="session_1")

    print("\n✅ Execution completed!")


if __name__ == "__main__":
    main()

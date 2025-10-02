import os
from typing import List
from dotenv import load_dotenv
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import tool

# Load environment variables
load_dotenv()

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI")

# Validate required variables
if not MONGODB_URI or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing required environment variables: MONGODB_URI or OPENAI_API_KEY")

# Connect to data
def init_mongodb():
    mongodb_client = MongoClient(MONGODB_URI)

    DB_NAME = "ai_agents"
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"]
    full_collection = mongodb_client[DB_NAME]["full_docs"]

    return mongodb_client, vs_collection, full_collection


def generate_embeddings(text: str) -> List[float]:
    """
    Generates embeddings for a piece of text.

    Args:
        text (str): The text to embed.
        embedding_model (text-embedding-3-small): The embedding model.

    Returns:
        List[float]: The embedding of the text.
    """

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    return embedding_model.embed_query(text)


@tool
def get_information_for_question_answering(user_query: str) -> str:
    """
    Retrieves relevant documents for a user query using vector search.

    Args:
        user_query (str): The user's query.

    Returns:
        str: The retrieved documents as a string.
    """

    query_embedding = generate_embeddings(user_query)
    vs_collection = init_mongodb()[1]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_search",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 5,
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
    context = "\n\n".join([doc.get("body") for doc in results])
    return context


@tool
def get_page_content_for_summarization(user_query: str) -> str:
    """
    Retrieves the content of a documentation page for summarization.

    Args:
        user_query (str): The user's query (title of the documentation page).

    Returns:
        str: The content of a documentation page.
    """

    full_collection = init_mongodb()[2]

    query = {"title": user_query}
    projection = {"_id": 0, "body": 1}

    document = full_collection.find_one(query, projection)
    if document:
        return document["body"]
    else:
        return "Document not found"


def main():
    mongodb_client, vs_collection, full_collection = init_mongodb()

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    tools = [
        get_information_for_question_answering,
        get_page_content_for_summarization,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "You are a helpful AI assistant."
                " You are provided with tools to answer questions and summarize technical documentation related to data."
                " Think step-by-step and use these tools to get the information required to answer the user query."
                " Do not re-run tools unless absolutely necessary."
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
                " You have access to the following tools: {tool_names}."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    bind_tools = llm.bind_tools(tools)

    llm_with_tools = prompt | bind_tools

    # For test if agent can invoke correct tools.
    # tool_call_check_1 = llm_with_tools.invoke(["What are some best practices for data backups in data?"]).tool_calls
    #
    # tool_call_check_2 = llm_with_tools.invoke(["Give me a ssummary of the page titled Create a data Deployment"]).tool_calls
    #
    # print("Tool 1 call check:")
    # print(tool_call_check_1)
    # print("Tool 1 call check:")
    # print(tool_call_check_2)

main()
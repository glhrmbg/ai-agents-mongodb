from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Setup
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["ai_agents"]
full_collection = db["full_docs"]
chunked_collection = db["chunked_docs"]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# Load dataset
# from https://huggingface.co/datasets/MongoDB/mongodb-docs/blob/main/mongodb_docs.json
docs = load_dataset("data/mongodb-docs")

# Insert full docs
for doc in docs["train"]:
    full_collection.insert_one(doc)

# Chunk, embed and insert
for doc in docs["train"]:
    body = doc.get("body", "")
    chunks = text_splitter.split_text(body)

    for chunk in chunks:
        # Create a new document keeping ALL original fields
        chunked_doc = dict(doc)  # Copy all fields
        chunked_doc["body"] = chunk  # Replace body with the chunk

        # Generate and add the embedding
        embedding = embeddings.embed_query(chunk)
        chunked_doc["embedding"] = embedding

        chunked_collection.insert_one(chunked_doc)

# Create vector index
chunked_collection.create_search_index({
    "name": "vector_index",
    "type": "vectorSearch",
    "definition": {
        "fields": [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,
            "similarity": "cosine"
        }]
    }
})

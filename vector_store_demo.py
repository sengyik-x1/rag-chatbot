# vector_store_demo.py

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection(name="my_documents")
except:
    pass

collection = chroma_client.create_collection(name="my_documents")

documents = [
    "Python is a popular programming language used in AI and data science.",
    "Machine learning models learn patterns from large datasets.",
    "RAG systems combine document retrieval with language model generation.",
    "Vector databases store embeddings and enable fast similarity search.",
    "The weather in Malaysia is hot and humid throughout the year.",
    "Nasi lemak is a traditional Malaysian dish with coconut rice.",
]

print("Storing documents in ChromaDB...")

# Embed ALL documents in one single API call — simpler and faster
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=documents        # pass the whole list at once
)

# Extract all embeddings from the response
embeddings = [item.embedding for item in response.data]

print(f"Got {len(embeddings)} embeddings")   # should print 6

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=embeddings,
    documents=documents
)

print(f"Stored {len(documents)} documents!")
print()

# Search
query = "How do RAG systems work?"

query_response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)
query_embedding = query_response.data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "distances"]
)

print(f"Query: '{query}'")
print()
print("Top 2 most relevant chunks:")
for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    similarity = 1 - dist
    print(f"  {i+1}. (score: {similarity:.4f}) {doc}")
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()

COLLECTION_NAME = "pdf_docs"   

def get_answer(question, n_chunks=3):
    question_emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = question_emb.data[0].embedding

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_chunks,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = results["documents"][0]

    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.
    Use ONLY the context below to answer the question. If the answer isn't in the context, say "I couldn't find that in the document."
Context:
{context}

Question: {question}
Answer:"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )


    answer = response.choices[0].message.content.strip()
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }

if __name__ == "__main__":
    print("RAG chatbot ready! Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nSearching documents...")
        result = get_answer(question)

        print(f"\nAnswer: {result['answer']}\n")
        print(f"\n--- Retrieved {len(result['retrieved_chunks'])} chunks ---")
        for i, chunk in enumerate(result['retrieved_chunks']):
            print(f"Chunk {i+1}: {chunk[:100]}...")
        print()


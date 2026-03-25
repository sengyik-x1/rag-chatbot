from openai import OpenAI
from dotenv import load_dotenv
import math

load_dotenv()
client = OpenAI()

def get_embeddings(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

question = "What animals make good pets?"

chunk1 = "Cats and dogs are the most popular household pets. They are friendly and loyal."
chunk2 = "The weather forecast shows rain for the next three days across the region."
chunk3 = "Rabbits and hamsters are also great pets for families with small children."

question_emb = get_embeddings(question)
chunk1_emb = get_embeddings(chunk1)
chunk2_emb = get_embeddings(chunk2)
chunk3_emb = get_embeddings(chunk3)

similarity1 = cosine_similarity(question_emb, chunk1_emb)
similarity2 = cosine_similarity(question_emb, chunk2_emb)
similarity3 = cosine_similarity(question_emb, chunk3_emb)

print(f"Question: '{question}'")
print()
print(f"Chunk 1 score: {similarity1:.4f} → '{chunk1[:50]}...'")
print(f"Chunk 2 score: {similarity2:.4f} → '{chunk2[:50]}...'")
print(f"Chunk 3 score: {similarity3:.4f} → '{chunk3[:50]}...'")
print()

scores = [(similarity1, chunk1), (similarity2, chunk2), (similarity3, chunk3)]
best_score, best_chunk = max(scores, key=lambda x: x[0])
print(f"Most relevant chunk: '{best_chunk}' with a score of {best_score:.4f}")
print(f"  → {best_chunk}")
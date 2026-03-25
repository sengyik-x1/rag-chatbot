from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

text1 = "I love cats and dogs"
text2 = "Kittens and puppies are adorable"
text3 = "The stock market crashed today"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

print(f"Embedding size: {len(emb1)} numbers")
print(f"First 5 numbers of embedding 1: {emb1[:5]}")
print()
print(f"Text 1: '{text1}'")
print(f"Text 2: '{text2}'")
print(f"Text 3: '{text3}'")

from openai import OpenAI
from dotenv import load_dotenv    
import os

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is RAG and how does it work?"
        }
    ]
)

print(response.choices[0].message.content)
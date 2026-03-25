from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
openai_client = OpenAI()

PDF_PATH = "ragpdf.pdf"
COLLECTION_NAME = "pdf_docs"

def ingest_pdf(pdf_path):
    print(f"Loading PDF from {pdf_path}...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from the PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    batch_size = 50
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk.page_content for chunk in batch]

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        collection.add(
            ids=[f"chunk_{j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page": chunk.metadata.get("page", 0)} for chunk in batch]
        )

        print(f" Embedded chunks {i+1} to {min(i+batch_size, total)} of {total}")

    print("Done!")
    return total

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file '{PDF_PATH}' not found.")
    else:
        ingest_pdf(PDF_PATH)
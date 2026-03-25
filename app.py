import streamlit as st
import chromadb
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "streamlit_docs"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False


def ingest_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(pages)

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

        collection.add(
            ids=[f"chunk_{j}" for j in range(len(batch))],
            embeddings=[item.embedding for item in response.data],
            documents=texts,
            metadatas=[{"page": chunk.metadata.get("page", 0)} for chunk in batch]
        )
    
    os.unlink(tmp_path)  # delete the tmp file
    return total

def get_answer(question):
    question_emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = question_emb.data[0].embedding

    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
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

    return response.choices[0].message.content

#---------UI Part------------

st.title("RAG Chatbot")
st.caption("Upload a PDF document and ask questions about its content!")

#upload area
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            total_chunks = ingest_pdf(uploaded_file)
            st.session_state.doc_loaded = True
            st.success(f"Document ingested with {total_chunks} chunks!")
    
    if st.session_state.doc_loaded:
        st.info("Document ready -ask questions below!")
    else:
        st.warning("Please upload a PDF document to start.")

#chat area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question about your document:"):
    if not st.session_state.doc_loaded:
        st.warning("Please upload a PDF document first!")
    
    else:

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching from provided document..."):
                answer = get_answer(question)
            
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


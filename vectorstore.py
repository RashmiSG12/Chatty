from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

def split_transcript(transcript: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([transcript])
    return chunks

# def create_vectorstore(chunks, persist_directory="chroma_db"):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
#     vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
#     return vectorstore

def create_vectorstore(chunks, video_id):
    persist_directory = f"chroma_db/{video_id}"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    if not os.path.exists(persist_directory):
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    else:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return vectorstore

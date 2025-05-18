from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["youtu.be"]:
        return parsed_url.path[1:]
    elif parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/embed/")[1]
    return None


def fetch_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print("Transcript loaded.")
        return transcript
    except TranscriptsDisabled:
        print("No transcript available for this video.")
        return None


def split_transcript(transcript: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([transcript])
    return chunks


def create_vectorstore(chunks, persist_directory="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore


def load_llm():
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.5,
        max_tokens=512
    )
    return llm


def create_qa_chain(llm, retriever):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


def main(youtube_url: str, query: str):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    transcript = fetch_transcript(video_id)
    if transcript is None:
        print("Exiting since no transcript is available.")
        return
    
    chunks = split_transcript(transcript)
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()
    qa_chain = create_qa_chain(llm, retriever)
    
    response = qa_chain.invoke(query)
    print("\nAnswer:", response)


if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=u47GtXwePms"
    query = "what are the key takeaways from the video?"
    main(youtube_url, query)

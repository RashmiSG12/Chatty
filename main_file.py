import config
from transcript import extract_video_id, fetch_transcript
from qa_chain import load_llm, create_qa_chain
from vectorstore import create_vectorstore, split_transcript

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
    youtube_url = "https://www.youtube.com/watch?v=LHXXI4-IEns&t=458s"
    query = "What is RNN?"
    main(youtube_url, query)
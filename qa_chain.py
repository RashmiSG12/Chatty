from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from prompt import prompts

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
    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                           retriever=retriever,
                                           chain_type = "stuff",
                                           chain_type_kwargs = {"prompt": prompts()})
    return qa_chain


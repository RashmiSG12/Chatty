from langchain.prompts import PromptTemplate

def prompts():
    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an intelligent AI assistant. Based on the following transcript content, answer the question in a helpful and concise way.
    Answer as a helpful assistant. If the question is not related to the context, respond with a just a 1 line polite reply.

    Transcript Context:
    {context}

    Question:
    {question}

    """
    )
    return prompt_template
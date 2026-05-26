from langchain_core.prompts import ChatPromptTemplate
from retrieval_pipeline import similarRetrieval
from model_utils.initialize_models import ModelLoader
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

def generateResponse(query, relevant_docs):
    """ 
    This function generates response using provided docs and user query
    """
    
    doc_text = chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])
    
    message = [
        SystemMessage(content = "You are a RAG agent"),
        HumanMessage(content="""Based on the following documents, answer the question.

        Question:
        {query}

        Documents:
        {docs}

        Please provide the answer based only on these documents.
        If the answer is not present, return "None"."""
        )
    ]
    
    # prompt = ChatPromptTemplate.from_template("""
    #     Based on the following documents, answer the question.

    #     Question:
    #     {query}

    #     Documents:
    #     {docs}

    #     Please provide the answer based only on these documents.
    #     If the answer is not present, return "None".
    #     """)
    
    model = ModelLoader.load_ollama_model()
    
    chain = message | model

    response = chain.invoke({
        "query": query,
        "docs": doc_text
    })
    
    return response


if __name__ == "__main__":
    query = "Does nature helps men?"
    relevant_docs = similarRetrieval(query)
    response = generateResponse(query, relevant_docs)
    print(response.content)
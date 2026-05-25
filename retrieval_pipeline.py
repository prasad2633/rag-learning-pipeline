from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def similarRetrieval(query, persist_db='db/chroma_db'):
    """This function creates embedding of query and does similarity search"""
    
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=persist_db,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
        )

    retriever = db.as_retriever(search_kwargs={'k': 3})
    
    relevant_docs = retriever.invoke(query)

    # print(f"User Query: {query}")

    # print("-----Context-----")
    # for i, doc in enumerate(relevant_docs,1):
    #     print(f"Document {i}: \n {doc.page_content}\n")

    return relevant_docs



if __name__ == "__main__":
    query = "Does nature helps men?"
    relevant_docs = similarRetrieval(query)
    doc_text = chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])
    
    prompt = ChatPromptTemplate.from_template("""
        Based on the following documents, answer the question.

        Question:
        {query}

        Documents:
        {docs}

        Please provide the answer based only on these documents.
        If the answer is not present, return "None".
        """)
    
    model = ChatOllama(
        model= "gemma3:latest",
        temperature=0
        )
    
    chain = prompt | model

    response = chain.invoke({
        "query": query,
        "docs": doc_text
    })
    
    print(response.content)
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from generation_pipeline import generateResponse
from model_utils.initialize_models import ModelLoader
from retrieval_pipeline import similarRetrieval

chat_history = []

def askQuestion(user_question):

    if chat_history:
        messages = [SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just rewritten question.")] + chat_history + [
            HumanMessage(content=f"New Question: {user_question}")
            ]
        model = ModelLoader.load_ollama_model()
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
        
    else:
        search_question = user_question
        print(f"Searching for: {search_question}")

    relevant_docs = similarRetrieval(search_question)
    
    result = generateResponse(search_question, relevant_docs)
    
    answer = result.content
    
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    
    return answer
        
def start_chat():
    print("Ask me a Question!, Type 'quit' to exit")

    while True:
        question = input("\n Your Question: ")

        if question.lower() == "quit":
            print("Good Bye!!")
            break

        askQuestion(question)

if __name__ == "__main__":
    start_chat()
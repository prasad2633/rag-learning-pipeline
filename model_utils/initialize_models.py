from langchain_ollama import ChatOllama

class ModelLoader:
    
    @staticmethod
    def load_ollama_model(model_name="gemma3:latest", temprature=0):
        """This function is used to call ollama models"""
        return ChatOllama(
        model= model_name,
        temperature=temprature
        )
        
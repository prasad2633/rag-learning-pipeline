from langchain_ollama import ChatOllama

class ModelLoader:
    
    @staticmethod
    def load_model(model_name="gemma3:latest", temprature=0):
        """This function is used to call models"""
        return ChatOllama(
        model= model_name,
        temperature=temprature
        )
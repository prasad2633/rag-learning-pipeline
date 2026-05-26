from langchain_ollama import ChatOllama
import boto3
from dotenv import load_dotenv

class ModelLoader:
    
    @staticmethod
    def load_ollama_model(model_name="gemma3:latest", temprature=0):
        """This function is used to call ollama models"""
        return ChatOllama(
        model= model_name,
        temperature=temprature
        )
        
    @staticmethod
    def load_claude_model():
        """This function is used to call claude models"""
        client = boto3.client(
            "bedrock-runtime",
            region_name="eu-north-1"
        )

        return client
        
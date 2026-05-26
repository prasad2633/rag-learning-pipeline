from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
    )
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker 
from model_utils.initialize_models import ModelLoader

class TextSplitterLoader:
    """
    This class contains few chunking methods implementation, which are being used in the project
    """
    
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def recursiveTextSplitter(self):
        """ This function helps create chunks using Recursive Character Text Splitting method"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
            )

    def characterTextSplitter(self):
        """This function helps create chunks using Character Text Splitting method"""
        return CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
            ) 

    @staticmethod
    def sematicTextSplitter():
        "This function helps create chunks using Semantic Text Splitting method where it uses ai embedding model"
        return SemanticChunker(
            embeddings=OllamaEmbeddings(model="nomic-embed-text"),
            breakpoint_threashold_type="percentile",
            breakpoint_threashold_amount=70
            )

    @staticmethod
    def agenticTextSplitter(raw_data):
        """ This function creates chunks using AI reasoning"""

        prompt = f"""
        You are a text chunking expert. Split this text into logical chunks.

        Rules:
        - Each chunk should be 200 characters or less.
        - Split at natural topic boundaries.
        - Keep related information together.
        - Put '<<<SPLIT>>>' between chunks.

        Text:
        {raw_data}

        Return the text with '<<<SPLIT>>>' markers where you want to split:
        """
        llm = ModelLoader.load_ollama_model()
        print("Asking AI to chunk the text...")
        response = llm.invoke(prompt)    
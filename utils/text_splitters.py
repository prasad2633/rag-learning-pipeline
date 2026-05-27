from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
    )
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker 
from model_utils.initialize_models import ModelLoader
from langchain_core.documents import Document

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

        Your task:
        Split the text into MULTIPLE chunks.

        STRICT RULES:
        1. You MUST split the text.
        2. Each chunk MUST be less than 500 characters.
        3. Never return the full text as one chunk.
        4. Split at natural topic or sentence boundaries.
        5. Keep related information together.
        6. Place <<<SPLIT>>> ONLY between chunks.
        7. Do not explain your reasoning.
        8. Output ONLY the chunked text.

        Text:
        {raw_data}

        Return the text with '<<<SPLIT>>>' markers where you want to split:
        """
        
        print("Asking AI to chunk the text...")
        
        llm = ModelLoader.load_ollama_model()
        response = llm.invoke(prompt)

        marked_text = response.content
        chunks = marked_text.split("<<<SPLIT>>>")

        print(chunks)

        # clean up white spaces
        cleaned_chunks = []
        try:
            for chunk in chunks:
                cleaned_ck = chunk.strip()
                
                if cleaned_ck:
                    cleaned_chunks.append(
                        Document(
                            page_content=cleaned_ck,
                            metadata={
                                "chunking_method": "agentic"
                                }
                            )
                        )

        except Exception as e:
            print(f"Error while cleaning the chunks:\n {e}")

        print("\n AGENTIC AI CHUNKING RESULTS:")    
        print("=" * 50)
        
        # testing the chunks:
        for i, chunk in enumerate(cleaned_chunks, 1):
            print(f"\nThe cleaned chunk {i}: {chunk}")

        return cleaned_chunks
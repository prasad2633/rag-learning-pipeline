import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.text_splitters import TextSplitterLoader

def docLoad(folder_path):
    """Load all documents from the directory"""
    print(f"Loading documents from {folder_path}...")
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The directory {folder_path} does not exists")

    loader = DirectoryLoader(
        path=folder_path,
        glob="*.txt",
        loader_cls=TextLoader
        )
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"The .txt file does no exists in the directory {folder_path}")

    return documents

def docChunk(documents, chunk_size = 100, chunk_overlap = 0):
    """Chunks the document"""
    print("Splitting documents in chunks")
    
    text_splitter = TextSplitterLoader.characterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    
    chunks = text_splitter.split_documents(documents)

    # chunks = TextSplitterLoader.agenticTextSplitter(documents)

    return chunks

def chunkEmbedding(chunks, persist_directory="db/chroma_db"):
    """Creates embeddings of the chunks"""
    
    print("Creating embeddings and storing it in chroma store...")
    
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    print("-------Creating Vector Store-------")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
        )

    print("-----Finished Creating Vector Store-----")

    print(f"Created vector store and saved it to {persist_directory}")
    return vectorstore

def main():
    # Step 1 : Load all the required text files
    folder_path = "./docs"
    documents = docLoad(folder_path)

    # step 2 : Chunk the documents 
    chunks = docChunk(documents)

    # step 3: Embedding 
    vectorstore = chunkEmbedding(chunks)
    
if __name__ == "__main__":
    main()
    
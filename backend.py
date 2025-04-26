import os
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

class DocumentQA:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        os.makedirs("data", exist_ok=True)
        
    def process_files(self):
        llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key)
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        return index.as_query_engine(llm=llm)
    
    def save_uploaded_files(self, uploaded_files):
        for file in uploaded_files:
            file_path = os.path.join("data", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        return len(uploaded_files)
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)

reader = SimpleDirectoryReader(input_dir="data")
documents = reader.load_data()

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("what is in pdf?")
print(response)
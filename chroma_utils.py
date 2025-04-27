import chromadb
import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List


class ChromaDocumentManager:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._index_cache = {}  # Cache for loaded indices
        os.makedirs(persist_dir, exist_ok=True)

    def load_and_index_document(self, file_path: str, file_id: int) -> bool:
        """Index a single document with metadata"""
        try:
            collection = self.chroma_client.get_or_create_collection(f"doc_{file_id}")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            for doc in documents:
                doc.metadata = {
                    "file_id": file_id,
                    "file_name": os.path.basename(file_path)
                }
            
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            return True
        except Exception as e:
            print(f"Error indexing {file_path}: {str(e)}")
            return False

    def get_query_engine(self, file_ids: List[int], llm=None):
        """Create query engine for multiple documents"""
        try:
            # Create multi-collection vector store
            vector_stores = []
            for file_id in file_ids:
                collection = self.chroma_client.get_collection(f"doc_{file_id}")
                vector_stores.append(ChromaVectorStore(chroma_collection=collection))
            
            storage_context = StorageContext.from_defaults(vector_store=vector_stores[0])
            if len(vector_stores) > 1:
                storage_context = StorageContext.from_defaults(
                    vector_stores=vector_stores
                )
                
            index = VectorStoreIndex.from_vector_store(
                vector_stores[0],
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            return index.as_query_engine(llm=llm)
        except Exception as e:
            print(f"Error creating query engine: {str(e)}")
            return None

    def clear_all(self):
        """Reset entire ChromaDB"""
        try:
            self._index_cache.clear()
            self.chroma_client.reset()
            return True
        except Exception as e:
            print(f"Error clearing ChromaDB: {str(e)}")
            return False  
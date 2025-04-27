from dotenv import load_dotenv
import os
from llama_index.llms.groq import Groq
from chroma_utils import ChromaDocumentManager

class DocumentQA:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self._ensure_data_folder()  # Auto-handles folder creation
        self.chroma_mgr = ChromaDocumentManager()
        self._current_file_ids = set()
        self._query_engine = None
        
    def _ensure_data_folder(self):
        """Automatically creates data folder if missing"""
        try:
            os.makedirs("data", exist_ok=True)  # No-op if exists
        except Exception as e:
            print(f"Error ensuring data folder: {e}")
            raise

    def process_files(self):
        """Process files with ChromaDB persistence"""
        if not self._query_engine and os.listdir("data"):  # Only if files exist
            llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key)
            
            for file_id, file_name in enumerate(os.listdir("data"), start=1):
                if file_id not in self._current_file_ids:
                    file_path = os.path.join("data", file_name)
                    if self.chroma_mgr.load_and_index_document(file_path, file_id):
                        self._current_file_ids.add(file_id)
            
            self._query_engine = self.chroma_mgr.get_query_engine(
                file_ids=list(self._current_file_ids),
                llm=llm
            )
        return self._query_engine
    
    def save_uploaded_files(self, uploaded_files):
        """Auto-creates data folder if missing during save"""
        
        for file in uploaded_files:
            file_path = os.path.join("data", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        return len(uploaded_files)
    
    def clear_data(self):
        """Clears all data including session tracking"""
        try:
            # Clear ChromaDB
            success = self.chroma_mgr.clear_all()
            
            # Clear local files
            for f in os.listdir("data"):
                os.remove(os.path.join("data", f))
            
            # Reset all tracking states
            self._current_file_ids.clear()
            self._query_engine = None
            
            # Add small delay for ChromaDB to fully reset
            import time
            time.sleep(1)
            
            return success
        except Exception as e:
            print(f"Error in clear_data: {str(e)}")
            return False
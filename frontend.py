import streamlit as st
from backend import DocumentQA

# Initialize backend
doc_qa = DocumentQA()

# Streamlit UI
st.title("📄 Document Q&A with Groq")
st.write("Upload files and ask questions about their content")

# File upload section
uploaded_files = st.file_uploader(
    "Choose files", 
    type=["pdf", "txt", "docx", "pptx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    # Save files using backend
    file_count = doc_qa.save_uploaded_files(uploaded_files)
    st.success(f"Uploaded {file_count} file(s) successfully!")
    
  
    query_engine = doc_qa.process_files()
    
    # Question input
    question = st.text_input("Ask a question about the uploaded documents:")
    
    if question:
        # Get and display response
        with st.spinner("Thinking..."):
            response = query_engine.query(question)
            st.write("### Answer")
            st.write(response.response)
        
        # Show source documents
        with st.expander("Show source documents"):
            for i, doc in enumerate(response.source_nodes):
                st.write(f"#### Document {i+1}")
                st.write(doc.text)
                st.write("---")
else:
    st.info("Please upload files to get started")
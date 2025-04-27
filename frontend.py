import streamlit as st
from backend import DocumentQA

# Initialize backend and session state
if 'doc_qa' not in st.session_state:
    st.session_state.doc_qa = DocumentQA()
    st.session_state.query_engine = None
    st.session_state.chat_history = []
    st.session_state.processed_files = set()  # Track processed files

def load_css(file_name):
    """Load CSS file from same directory"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize app with CSS
load_css("styles.css") 

# Streamlit UI
st.title("📚 Smart Document Analyzer")
st.markdown("Upload documents and get intelligent answers from their content")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Choose file", 
        type=["pdf", "txt", "docx", "pptx", "csv"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        # Get current file names
        current_files = {hash(f.getvalue()) for f in uploaded_files}
        
        # Only process if these are new files
        if current_files != st.session_state.processed_files:
            with st.spinner("Processing documents..."):
                try:

                     # Clear previous data
                    st.session_state.doc_qa.clear_data()
                    st.session_state.processed_files = set()

                    file_count = st.session_state.doc_qa.save_uploaded_files(uploaded_files)
                    query_engine = st.session_state.doc_qa.process_files()
                    
                    if query_engine:
                        st.session_state.query_engine = query_engine
                        st.session_state.processed_files = current_files
                        st.success(f"Processed {file_count} file(s)")
                        
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    

# Main chat interface
if st.session_state.query_engine:
    st.header("Chat with Your Documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            if message['role'] == 'user':
                st.markdown(
                    f"""
                    <div class='chat-message user-message'>
                        <span class='user-avatar'>👤</span>
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='chat-message bot-message'>
                        <span class='bot-avatar'>🤖</span>
                        <strong>Assistant:</strong> {message['content']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                with st.expander("🔍 View Sources & References"):
                    for i, doc in enumerate(message['sources']):
                        st.markdown(f"**Source {i+1}** (Score: {doc.score:.2f})")
                        st.caption(doc.text)
                        st.markdown("---")
    
    # Query input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user question to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question,
            'sources': []
        })
        
        # Get and display response
        with st.spinner("Analyzing documents..."):
            response = st.session_state.query_engine.query(question)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response.response,
                'sources': response.source_nodes
            })
            
            # Rerun to update the chat display
            st.rerun()
else:
    st.info("Please upload documents to begin")

# Add document count in sidebar
if st.session_state.query_engine:
    st.sidebar.markdown("---")
    st.sidebar.metric("Chat Messages", len(st.session_state.chat_history))
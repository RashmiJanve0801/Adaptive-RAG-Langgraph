import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import modular components
from src.LLMs.geminillm import initialize_llm_components
from src.Graphs.graph_builder import build_graph

# Set page config
st.set_page_config(page_title="Adaptive RAG System", page_icon="🤖", layout="wide")

# Load environment variables
load_dotenv()

# Function to build the index (defined early so sidebar can use it)
def build_index(urls):
    """Build index from provided URLs"""
    try:
        # Set embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load documents
        docs = []
        progress_bar = st.progress(0)
        for i, url in enumerate(urls):
            try:
                st.text(f"Loading: {url}")
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                progress_bar.progress((i + 1) / len(urls))
            except Exception as e:
                st.warning(f"Failed to load {url}: {str(e)}")
        
        progress_bar.empty()
        
        if not docs:
            st.error("No documents were successfully loaded!")
            return None
        
        st.text(f"Loaded {len(docs)} documents. Splitting text...")
        
        # Split - using standard splitter without tiktoken
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        doc_splits = text_splitter.split_documents(docs)
        
        st.text(f"Created {len(doc_splits)} chunks. Building vector store...")
        
        # Add to vectorstore
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=embeddings
        )
        
        retriever = vectorstore.as_retriever()
        st.text("Vector store built successfully!")
        return retriever
        
    except Exception as e:
        st.error(f"Error building index: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .debug-info {
        font-size: 0.8rem;
        color: #666;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stProgress > div > div {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for API keys and URLs
with st.sidebar:
    st.title("🔧 Configuration")
    
    with st.expander("API Keys", expanded=False):
        google_api_key = st.text_input("Google API Key", 
                                    value=os.getenv("GOOGLE_API_KEY", ""), 
                                    type="password")
        tavily_api_key = st.text_input("Tavily API Key", 
                                     value=os.getenv("TAVILY_API_KEY", ""),
                                     type="password")
        
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
    
    st.markdown("---")
    
    with st.expander("📚 Vector Store URLs", expanded=True):
        st.markdown("**Add URLs to build your knowledge base:**")
        
        # Default URLs
        default_urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        
        # Initialize session state for URLs
        if "custom_urls" not in st.session_state:
            st.session_state.custom_urls = default_urls.copy()
        
        # Display current URLs with delete buttons
        urls_to_remove = []
        for i, url in enumerate(st.session_state.custom_urls):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text_input(f"URL {i+1}", value=url, key=f"url_display_{i}", disabled=True)
            with col2:
                if st.button("🗑️", key=f"delete_{i}"):
                    urls_to_remove.append(i)
        
        # Remove URLs marked for deletion
        for idx in sorted(urls_to_remove, reverse=True):
            st.session_state.custom_urls.pop(idx)
            st.rerun()
        
        # Add new URL
        new_url = st.text_input("Add new URL", key="new_url_input", placeholder="https://example.com/article")
        if st.button("➕ Add URL"):
            if new_url and new_url.strip():
                if new_url.strip() not in st.session_state.custom_urls:
                    st.session_state.custom_urls.append(new_url.strip())
                    st.rerun()
                else:
                    st.warning("This URL is already in the list!")
            else:
                st.warning("Please enter a valid URL!")
        
        # Reset to defaults
        if st.button("🔄 Reset to Default URLs"):
            st.session_state.custom_urls = default_urls.copy()
            # Clear the cached index
            if "retriever" in st.session_state:
                del st.session_state.retriever
            st.rerun()
        
        # Build index button
        if st.button("🔨 Build/Rebuild Vector Store", type="primary"):
            if st.session_state.custom_urls:
                with st.spinner("Building vector store..."):
                    retriever = build_index(st.session_state.custom_urls)
                    if retriever:
                        st.session_state.retriever = retriever
                        st.success(f"✅ Vector store built with {len(st.session_state.custom_urls)} URLs!")
                    else:
                        st.error("Failed to build vector store. Check the URLs and try again.")
            else:
                st.warning("Please add at least one URL!")
    
    st.markdown("---")
    
    st.markdown("""
    ### About This App
    
    This app uses an adaptive RAG system that:
    
    1. 🧠 Routes your question to the best data source
    2. 🔍 Retrieves relevant documents
    3. ✅ Grades document relevance
    4. 🔄 Transforms queries when needed
    5. 📊 Verifies answers are factual
    
    Built with LangChain, LangGraph and Google Gemini.
    """)

# Main app
st.title("🤖 Adaptive RAG System")
st.markdown("Ask a question and the system will adaptively decide the best approach to answer it.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize components (cached)
@st.cache_resource
def get_components():
    return initialize_llm_components()

components = get_components()

# Display info message if vector store not built
if st.session_state.retriever is None:
    st.info("👉 Please build the vector store in the sidebar before asking questions!")

# Input for question
question = st.chat_input("Ask a question about AI, agents, prompt engineering, etc.")

if question:
    # Check if API keys are set
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        with st.chat_message("assistant"):
            st.warning("⚠️ Please set the Google API Key and Tavily API Key in the sidebar to use this app.")
        st.stop()
    
    # Check if vector store is built
    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please build the vector store first by clicking '🔨 Build/Rebuild Vector Store' in the sidebar!")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.debug_info = []
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Build and run workflow
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build workflow with components
                workflow = build_graph(components)
                
                # Run the workflow
                result = workflow.invoke({"question": question})
                
                # Extract answer
                answer = result.get("generation", "I couldn't find an answer to your question.")
                
                # Display answer
                st.markdown(answer)
                
                # Display debug info if requested
                with st.expander("Show processing steps"):
                    for item in st.session_state.debug_info:
                        st.markdown(f"- {item}")
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
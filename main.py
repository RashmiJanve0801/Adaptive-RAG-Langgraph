import os
import streamlit as st
from dotenv import load_dotenv
import certifi

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph imports
from langgraph.graph import END, StateGraph, START
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Set page config
st.set_page_config(page_title="Adaptive RAG System", page_icon="🤖", layout="wide")

# Load environment variables
load_dotenv()

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

# Sidebar for API keys
with st.sidebar:
    st.title("🔑 API Configuration")
    
    with st.expander("Configure API Keys", expanded=False):
        google_api_key = st.text_input("Google API Key", 
                                    value=os.environ.get("GOOGLE_API_KEY", ""), 
                                    type="password")
        tavily_api_key = st.text_input("Tavily API Key", 
                                     value=os.environ.get("TAVILY_API_KEY", ""),
                                     type="password")
        
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
    
    st.markdown("---")
    
    st.markdown("""
    ### About This App
    
    This app uses an adaptive RAG system that:
    
    1. 🧠 Routes your question to the best data source
    2. 📝 Retrieves relevant documents
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

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to build the index
@st.cache_resource
def build_index():
    # Set embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Docs to index
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    try:
        # Load
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        # Split
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        # Add to vectorstore
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=embeddings
        )
        
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"Error building index: {str(e)}")
        return None

# Set up data models for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

# Initialize LLM and tools
@st.cache_resource
def initialize_components():
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)
    structured_llm_grader_hall = llm.with_structured_output(GradeHallucinations)
    structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
    
    # Web search
    web_search_tool = TavilySearchResults(k=3)
    
    # Prompts
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""),
        ("human", "{question}"),
    ])
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    # RAG prompt
    rag_prompt = hub.pull("rlm/rag-prompt")
    
    question_router = route_prompt | structured_llm_router
    retrieval_grader = grade_prompt | structured_llm_grader_docs
    hallucination_grader = hallucination_prompt | structured_llm_grader_hall
    answer_grader = answer_prompt | structured_llm_grader_answer
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    
    # RAG chain
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    return {
        "llm": llm,
        "question_router": question_router,
        "retrieval_grader": retrieval_grader,
        "hallucination_grader": hallucination_grader,
        "answer_grader": answer_grader,
        "question_rewriter": question_rewriter,
        "rag_chain": rag_chain,
        "web_search_tool": web_search_tool
    }

# Get components
components = initialize_components()
retriever = build_index()

# Graph state functions
def retrieve(state):
    """Retrieve documents"""
    st.session_state.debug_info.append("📚 Retrieving documents from vector store")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """Generate answer"""
    st.session_state.debug_info.append("✍️ Generating answer")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = components["rag_chain"].invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """Determines whether the retrieved documents are relevant to the question."""
    st.session_state.debug_info.append("🔍 Checking document relevance")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = components["retrieval_grader"].invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            st.session_state.debug_info.append("✅ Document is relevant")
            filtered_docs.append(d)
        else:
            st.session_state.debug_info.append("❌ Document is not relevant")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """Transform the query to produce a better question."""
    st.session_state.debug_info.append("🔄 Transforming query")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = components["question_rewriter"].invoke({"question": question})
    st.session_state.debug_info.append(f"📝 Transformed query: {better_question}")
    return {"documents": documents, "question": better_question}


def web_search(state):
    """Web search based on the re-phrased question."""
    st.session_state.debug_info.append("🔎 Performing web search")
    question = state["question"]

    # Web search
    docs = components["web_search_tool"].invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    
    from langchain.schema import Document
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


# Edge functions
def route_question(state):
    """Route question to web search or RAG."""
    st.session_state.debug_info.append("🧭 Routing question")
    question = state["question"]
    source = components["question_router"].invoke({"question": question})
    if source.datasource == "web_search":
        st.session_state.debug_info.append("📊 Routing to web search")
        return "web_search"
    elif source.datasource == "vectorstore":
        st.session_state.debug_info.append("📚 Routing to vectorstore")
        return "vectorstore"


def decide_to_generate(state):
    """Determines whether to generate an answer, or re-generate a question."""
    st.session_state.debug_info.append("🤔 Deciding whether to generate answer")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        st.session_state.debug_info.append("📝 All documents are not relevant, transforming query")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        st.session_state.debug_info.append("✅ Documents are relevant, generating answer")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """Determines whether the generation is grounded in the document and answers question."""
    st.session_state.debug_info.append("🧐 Checking for hallucinations")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = components["hallucination_grader"].invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        st.session_state.debug_info.append("✅ Answer is grounded in documents")
        # Check question-answering
        st.session_state.debug_info.append("🧐 Checking if answer addresses question")
        score = components["answer_grader"].invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            st.session_state.debug_info.append("✅ Answer addresses question")
            return "useful"
        else:
            st.session_state.debug_info.append("❌ Answer does not address question")
            return "not useful"
    else:
        st.session_state.debug_info.append("❌ Answer is not grounded in documents, retrying")
        return "not supported"

# Build workflow
@st.cache_resource
def build_workflow():
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    
    # Compile
    app = workflow.compile()
    return app

# Input for question
question = st.chat_input("Ask a question about AI, agents, prompt engineering, etc.")

if question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.debug_info = []
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Check if API keys are set
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        with st.chat_message("assistant"):
            st.warning("Please set the Google API Key and Tavily API Key in the sidebar to use this app.")
        st.stop()
    
    # Get workflow
    workflow = build_workflow()
    
    # Process with progress bar
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
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

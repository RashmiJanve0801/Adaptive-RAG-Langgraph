import streamlit as st
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize web search tool
web_search_tool = TavilySearchResults(k=3)

def retrieve(state, components):
    """Retrieve documents from vector store"""
    st.session_state.debug_info.append("📚 Retrieving documents from vector store")
    question = state["question"]

    # Retrieval
    retriever = st.session_state.retriever
    if not retriever:
        st.session_state.debug_info.append("❌ Vector store not initialized!")
        return {"documents": [], "question": question}
    
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state, components):
    """Generate answer using RAG"""
    st.session_state.debug_info.append("✏️ Generating answer")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = components["rag_chain"].invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state, components):
    """Grade retrieved documents for relevance"""
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


def transform_query(state, components):
    """Transform the query to produce a better question"""
    st.session_state.debug_info.append("🔄 Transforming query")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = components["question_rewriter"].invoke({"question": question})
    st.session_state.debug_info.append(f"🔍 Transformed query: {better_question}")
    return {"documents": documents, "question": better_question}


def web_search(state, components):
    """Perform web search"""
    st.session_state.debug_info.append("🔎 Performing web search")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


def route_question(state, components):
    """Route question to web search or RAG"""
    st.session_state.debug_info.append("🧭 Routing question")
    question = state["question"]
    source = components["question_router"].invoke({"question": question})
    if source.datasource == "web_search":
        st.session_state.debug_info.append("📊 Routing to web search")
        return "web_search"
    elif source.datasource == "vectorstore":
        st.session_state.debug_info.append("📚 Routing to vectorstore")
        return "vectorstore"


def decide_to_generate(state, components):
    """Decide whether to generate an answer or transform query"""
    st.session_state.debug_info.append("🤔 Deciding whether to generate answer")
    filtered_documents = state["documents"]

    if not filtered_documents:
        st.session_state.debug_info.append("🔍 All documents are not relevant, transforming query")
        return "transform_query"
    else:
        st.session_state.debug_info.append("✅ Documents are relevant, generating answer")
        return "generate"


def grade_generation_v_documents_and_question(state, components):
    """Check if generation is grounded and answers the question"""
    st.session_state.debug_info.append("🧠 Checking for hallucinations")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = components["hallucination_grader"].invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        st.session_state.debug_info.append("✅ Answer is grounded in documents")
        # Check question-answering
        st.session_state.debug_info.append("🧠 Checking if answer addresses question")
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
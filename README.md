# Adaptive RAG System using LangGraph

Most RAG systems are overconfident. They retrieve whatever's closest in 
vector space, stuff it into a prompt, and generate an answer — even if 
the retrieved docs have nothing to do with your question.

This one doesn't do that. It checks its own work before answering.

## What makes it "adaptive"

Three things most RAG systems skip:

1. **Document grading** — after retrieval, it scores whether the docs 
   are actually relevant to the question. If they're not, it doesn't 
   just give up.

2. **Query rewriting** — if retrieval quality is low, it rewrites 
   your original question to be more precise, then tries again.

3. **Hallucination detection** — before returning an answer, it verifies 
   the response is grounded in what was retrieved, not invented.

The Streamlit UI shows every decision in real time — which path it took, 
why it rewrote the query, whether it flagged hallucination. 
I added that for debugging and kept it because it's genuinely interesting.


## How the workflow runs

```
Query
  → Router: vector DB or web search?
  → Retrieve docs
  → Grade relevance → if poor: rewrite query → re-retrieve
  → Generate answer
  → Hallucination check → if flagged: regenerate
  → Return verified answer
```

## Deployment

This isn't just a local Streamlit app. It's running on GCP with a 
full GitOps CI/CD pipeline:

- **Jenkins** builds on every GitHub push via webhook
- **ArgoCD** handles continuous delivery to GKE
- Manual deploys used to take ~20 mins. Now it's under 5, automatically.

## Tech Stack

## Tech Stack

| Layer | Technologies |
|---|---|
| Agent Orchestration | LangGraph, LangChain |
| LLM & Embeddings | Google Gemini Pro/Flash, FAISS |
| Retrieval | FAISS (vector store), Tavily (web search) |
| Frontend | Streamlit |
| CI/CD | Jenkins, ArgoCD, GitHub Webhooks |
| Infra | Docker, GKE (Google Kubernetes Engine), python-dotenv |



# 🔍 Adaptive RAG System using LangGraph

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

## Live Demo
👉 [Try it here](your-demo-link-here)
🔗 [GitHub](your-github-link-here)

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

## Stack

| What | Why |
|---|---|
| LangGraph | Workflow orchestration + stateful graph |
| LangChain | LLM tooling + prompt management |
| Google Gemini | Generation + document grading |
| FAISS | Local vector store |
| Tavily | Web search fallback |
| Streamlit | Frontend UI |
| Docker | Containerized app |
| GKE (Google Kubernetes Engine) | Cloud deployment |
| Jenkins | CI pipeline (builds + tests) |
| ArgoCD | CD pipeline (GitOps deployments) |

## Run locally

```bash
git clone https://github.com/rashmijanve/adaptive-rag-langgraph.git
cd adaptive-rag-langgraph
pip install -r requirements.txt
```

`.env`:
```env
GEMINI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

```bash
streamlit run app.py
```

For the full Kubernetes setup, check the `/k8s` folder —
all manifests are there with comments.

## Project structure

```
adaptive-rag-langgraph/
├── graph/
│   ├── nodes.py        # retrieval, grading, generation, hallucination check
│   ├── edges.py        # conditional routing logic
│   └── workflow.py     # LangGraph graph assembly
├── retriever/
│   ├── vector_store.py # FAISS setup + document ingestion
│   └── web_search.py   # Tavily integration
├── k8s/                # GKE deployment manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── Jenkinsfile         # CI pipeline definition
├── app.py
├── requirements.txt
└── README.md
```

## What I'd improve next

- Swap FAISS for a managed vector DB (Pinecone or AstraDB) for scale
- Add feedback loop — let users flag bad answers to improve retrieval
- Experiment with reranking before generation

## License
MIT

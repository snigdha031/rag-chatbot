# PDF RAG Chatbot

**Live Demo:** 
👉 https://rag-chatbot-otcfoy7nvekby3tmmxycba.streamlit.app/

🎥 Demo [demo.gif]

A conversational AI chatbot that lets you upload PDFs and ask questions about them. Built with Retrieval-Augmented Generation (RAG), it retrieves the most relevant chunks from your documents, reranks them for accuracy, and generates structured answers using LLaMA 3.3 70B via Groq.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.x-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Features

- **Multi-PDF support** — upload and index multiple PDFs in a single session
- **Smart Mode** — answers using document context, falls back to general knowledge when needed
- **Strict Mode** — answers strictly from the uploaded documents, no external knowledge
- **Reranking pipeline** — BGE cross-encoder reranker on top of MMR retrieval for higher precision
- **Streaming responses** — token-by-token output with a live typing cursor
- **Query-type detection** — automatically detects compare, summarize, definition, explain, or general queries and formats the answer accordingly
- **Source attribution** — shows which file and page each answer was pulled from, grouped by document
- **Chat export** — download the full conversation as a `.txt` file

---

## How It Works

```
User Query
    |
Query Classifier --> Format Instructions
    |
History-Aware Query Rewriter  (LLaMA 3.3 via Groq)
    |
MMR Retriever  (ChromaDB + all-MiniLM-L6-v2)
fetch_k=40 --> k=20
    |
BGE Reranker  (BAAI/bge-reranker-base)
    |
Top 5 Chunks
    |
Answer Generation  (LLaMA 3.3 70B via Groq)
    |
Streaming Response + Source Expander
```

1. **Ingest** — PDFs are loaded with `PyPDFLoader`, split into 800-token chunks with 150-token overlap, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB.

2. **Retrieve** — On each query, chat history is used to rewrite the question into a clear standalone query. MMR retrieval fetches 40 candidates, filters to 20 for diversity, then the BGE cross-encoder reranks and returns the top 5.

3. **Generate** — The top chunks are passed to LLaMA 3.3 70B via Groq along with the detected query type and structured format instructions. The response streams token by token.

4. **Sources** — Retrieved chunks are grouped by source file and shown with page numbers and text excerpts in a collapsible expander.

---

## Project Structure

```
rag-chatbot/
├── app.py              # Streamlit UI — chat interface, sidebar, export
├── rag_chain.py        # RAG pipeline — retriever, reranker, LLM chain
├── ingest.py           # PDF ingestion — load, chunk, embed, store
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/snigdha031/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| Reranker model | `BAAI/bge-reranker-base` | Cross-encoder reranker |
| LLM | `llama-3.3-70b-versatile` | Via Groq API |
| Chunk size | `800` tokens | With `150` token overlap |
| MMR fetch_k | `40` | Candidates fetched before MMR |
| MMR k | `20` | After MMR diversity filter |
| Reranker top_k | `5` | Final chunks passed to LLM |

---

## Deployment

### Streamlit Community Cloud (free)

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add your API key under **Settings > Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

5. Click Deploy — your app gets a public URL instantly

> **Note:** ChromaDB persists to disk. On Streamlit Cloud the filesystem resets on each reboot, so uploaded PDFs will need to be re-indexed after the app restarts. For persistent storage across restarts, swap ChromaDB for a hosted vector DB like Pinecone or Qdrant Cloud.

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | LLaMA 3.3 70B via Groq |
| Orchestration | LangChain |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector store | ChromaDB |
| Reranker | BAAI/bge-reranker-base |
| PDF loading | PyPDFLoader |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

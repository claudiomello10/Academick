# AcademiCK — Simplified CLI Version

This is a **simplified version of AcademiCK for educational purposes**. It is designed for students who want to **understand how the RAG pipeline works** without dealing with Docker, microservices, or databases.

> **Note:** This is NOT the production version. For the full application with web interface, authentication, and scalable architecture, see the [main project](https://github.com/claudiomello10/Academick). This version exists purely as a learning tool.

It does the exact same thing as the full app — processes PDFs, searches through them using AI, and answers your questions with citations — but in ~600 lines of Python across 4 files.

## How It Works

The app follows a **Retrieval-Augmented Generation (RAG)** pipeline. Here's what happens when you ask a question:

```
+---------------------+
|   Your Question     |
+----------+----------+
           |
           v
+---------------------+
| 1. Intent           |  Classifies your question into one of:
|    Classification    |  Q&A | Summarization | Coding | Search
+----------+----------+
           |
           v
+---------------------+
| 2. Query            |  LLM generates 3 optimized
|    Enhancement      |  search queries from your question
+----------+----------+
           |
           v
+---------------------+
| 3. Hybrid Search    |  Finds the most relevant text chunks:
|                     |  - Dense search  (semantic meaning)
|                     |  - Sparse search (keyword matching)
|                     |  Combined with intent-based weights
+----------+----------+
           |
           v
+---------------------+
| 4. Prompt           |  Builds a system prompt with the
|    Engineering      |  retrieved context + intent instructions
+----------+----------+
           |
           v
+---------------------+
| 5. LLM Response     |  Generates the final answer
|                     |  with book/chapter/section citations
+---------------------+
```

## File Structure

```
simple/
├── main.py              # CLI menu — start here
├── student_helper.py    # Core RAG class — the brain
├── pdf_processor.py     # PDF processing — turning books into searchable chunks
├── llm_provider.py      # LLM interface — talks to OpenAI/Anthropic/DeepSeek
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration template
└── README.md            # You are here
```

### What each file does

#### `llm_provider.py` (~110 lines)

The simplest file. It provides a single function `generate_response(messages, model)` that routes your LLM call to the correct provider based on the model name:

- `gpt-*` → OpenAI
- `claude-*` → Anthropic
- `deepseek-*` → DeepSeek

#### `pdf_processor.py` (~220 lines)

Turns PDF textbooks into searchable chunks:

1. **Extracts the Table of Contents** from the PDF using PyMuPDF
2. **Identifies chapters** by asking an LLM to analyze the TOC
3. **Extracts text** page by page, organized by chapter and topic
4. **Chunks the text** into ~3000 character pieces with 1000 character overlap (so no information is lost at boundaries)
5. **Filters out** short chunks (<300 chars) and index/TOC pages (too many dots)
6. **Generates dense embeddings** for each chunk using the BGE-M3 model (1024-dimensional vectors that capture semantic meaning)

The result is a pandas DataFrame with columns: `Book, Chapter, Text, Topic, is_introduction, dense_embedding`

#### `student_helper.py` (~330 lines)

The core of the app. The `StudentHelper` class handles:

**Model Loading** — Loads two ML models at startup:

- **BGE-M3** (`BAAI/bge-m3`): Generates embeddings (numerical representations) of text for semantic search
- **Intent Classifier** (`claudiomello/AcademiCK-intent-classifier`): Determines what kind of question you're asking

**Hybrid Search** — A two-stage process to find the most relevant chunks:

1. **Dense pre-filter**: Computes cosine similarity between your query embedding and all stored chunk embeddings using a fast numpy dot product. Keeps the top 50 candidates.
2. **Sparse rerank**: For those 50 candidates, computes sparse (keyword-based) similarity scores. Then combines dense + sparse scores with weights that depend on the intent:
   - Q&A: 60% dense, 40% sparse (balance of meaning + keywords)
   - Summarization: 70% dense, 30% sparse (meaning matters more)
   - Coding: 40% dense, 60% sparse (keywords matter more for code)
   - Search: 50/50

**Query Enhancement** — Before searching, the app asks an LLM to generate 3 optimized search queries from your original question. This improves retrieval because a single question can be searched from multiple angles.

**Prompt Engineering** — 4 different system prompts tailored to each intent type. For example, the Q&A prompt instructs the LLM to never give direct answers to exercises — instead guiding the student to think through the problem.

**Conversation History** — Keeps track of your chat so follow-up questions work naturally.

#### `main.py` (~160 lines)

The CLI interface. Provides a simple menu:

1. **Chat** — Ask questions and get RAG-powered answers
2. **Process PDFs** — Turn PDF textbooks into searchable embeddings
3. **List books** — See what's been processed
4. **Change subject** — Switch the study topic
5. **Change model** — Switch LLM provider/model
6. **Clear history** — Reset conversation
7. **Exit**

## Quick Start

### Prerequisites

- Python 3.10+
- At least one LLM API key (OpenAI, Anthropic, or DeepSeek)
- 4GB+ free disk space (for ML model downloads)
- 8GB+ RAM recommended

### Setup

```bash
cd simple

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your API key
```

### Run

```bash
python main.py
```

On the first run, the app will download two ML models (~2 GB total):

- `BAAI/bge-m3` — the embedding model
- `claudiomello/AcademiCK-intent-classifier` — the intent classifier

These are cached locally by HuggingFace and won't be downloaded again.

### Usage

1. **Process a PDF first** (option 2) — point it to a PDF file or a directory of PDFs
2. **Start chatting** (option 1) — ask questions about the content
3. The app will classify your intent, search the books, and generate a cited answer

## How It Compares to the Full App

| Aspect                      | Full App (Docker)             | Simple CLI                       |
| --------------------------- | ----------------------------- | -------------------------------- |
| **Architecture**      | 9 microservices               | 4 Python files                   |
| **Vector Database**   | Qdrant                        | numpy dot products in memory     |
| **Metadata Database** | PostgreSQL                    | Not needed                       |
| **Cache**             | Redis                         | Not needed                       |
| **Task Queue**        | Celery workers                | Direct function calls            |
| **Embeddings**        | HTTP API to embedding service | Direct model loading             |
| **Intent**            | HTTP API to intent service    | Direct model loading             |
| **Storage**           | Qdrant + PostgreSQL           | pandas DataFrame + pickle file   |
| **Web Interface**     | Next.js frontend              | Terminal input/output            |
| **Auth**              | Session-based login           | None                             |
| **Concurrency**       | Async everywhere              | Synchronous                      |
| **PDF Processing**    | Background Celery jobs        | Inline, blocking                 |
| **Setup**             | `docker compose up`         | `pip install + python main.py` |

The core RAG logic is identical — same prompts, same search weights, same query enhancement, same intent classification.

## Configuration

All settings are in the `.env` file:

| Variable                    | Default             | Description                                    |
| --------------------------- | ------------------- | ---------------------------------------------- |
| `OPENAI_API_KEY`          | —                  | OpenAI API key                                 |
| `ANTHROPIC_API_KEY`       | —                  | Anthropic API key                              |
| `DEEPSEEK_API_KEY`        | —                  | DeepSeek API key                               |
| `DEFAULT_MODEL`           | gpt-5-mini          | LLM for final responses                        |
| `QUERY_ENHANCEMENT_MODEL` | gpt-5-nano          | LLM for generating search queries (cheap/fast) |
| `DEFAULT_SUBJECT`         | Machine Learning    | Study subject context                          |
| `EMBEDDING_MODEL`         | BAAI/bge-m3         | HuggingFace model for embeddings               |
| `DEVICE`                  | cpu                 | `cpu` or `cuda` for GPU acceleration       |
| `DATA_PATH`               | data/embeddings.pkl | Where processed embeddings are stored          |
| `PDF_DIRECTORY`           | books/              | Default directory for PDF files                |

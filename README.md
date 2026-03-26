# Automated Research Agent

A multi-agent financial analysis system based on LangGraph, integrating Text2SQL, code interpreter, PDF parsing, web search, and RAG retrieval capabilities.

## ✨ Features

| Feature                 | Description                                                             |
|-------------------------|-------------------------------------------------------------------------|
| 🔍 **Text2SQL**         | Convert natural language to SQL for intelligent queries                 |
| 💻 **Code Interpreter** | Securely execute Python code for data analysis and visualization        |
| 📄 **File Parsing**     | Parse research reports and extract key information                      |
| 🌐 **Web Search**       | Retrieve real-time market trends and news                               |
| 📚 **RAG Retrieval**    | Vector-based semantic search for deep understanding of research reports |
| 🔄 **ReAct Reflection** | Multi-turn reasoning mechanism with self-evaluation and improvement     |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 LangGraph Orchestrator                     │  │
│  │                                                           │  │
│  │   Router ──▶ Planner ──▶ Executor ──▶ Reflector ──▶ Critic│  │
│  │     │                       │            │                │  │
│  │     └───────────────────────┼────────────┘                │  │
│  │                             ▼                             │  │
│  │   ┌─────────────────────────────────────────────────┐    │  │
│  │   │               Tool Registry                      │    │  │
│  │   ├─────────┬─────────┬─────────┬─────────┬────────┤    │  │
│  │   │Text2SQL │  Code   │  File   │   Web   │  RAG   │    │  │
│  │   │  Tool   │Executor │ Parser  │ Search  │ Search │    │  │
│  │   └─────────┴─────────┴─────────┴─────────┴────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│           ▼           ▼           ▼           ▼                 │
│     [ SQLite ]  [ Python ]  [ DocMind ]  [ Bocha API ]            │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd AutoResearcher
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Alibaba Cloud DashScope API Key (for qwen-max and text-embedding-v4)
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

### 3. Start Service

```bash
python main.py
```

After startup, access:
- **API Docs:**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
Agent3/
├── main.py                 # FastAPI entry point
├── config.py               # Configuration file
├── requirements.txt        # Dependency list
│
├── database/               # Database module
│   └── init_db.py          # Database initialization
│
├── agents/                 # LangGraph agents
│   ├── nodes.py            # Agent node implementations
│   └── graph.py            # Workflow definition
│
├── tools/                  # Toolset
│   ├── text2sql.py         # Natural language to SQL
│   ├── code_executor.py    # Python code executor
│   ├── file_parser.py      # File parsing tool
│   ├── web_search.py       # Web search (Bocha API)
│   └── rag_search.py       # RAG semantic retrieval
│
├── services/               # Service layer
│   ├── llm.py              # LLM service (qwen-max)
│   └── embedding.py        # Embedding service
│
└── api/                    # API routers
    └── routers.py          # FastAPI router definitions
```


## 🔌 API Endpoints

### Core Endpoints

| Endpoint           | Method | Description                   |
|--------------------|--------|-------------------------------|
| `/api/chat`        | POST   | Intelligent chat (main entry) |
| `/api/chat/stream` | POST   | Streaming chat                |

### Tool Endpoints

| Endpoint            | Method | Description         |
|---------------------|--------|---------------------|
| `/api/sql/query`    | POST   | Text2SQL query      |
| `/api/code/execute` | POST   | Execute Python code |
| `/api/search`       | POST   | Web search          |
| `/api/rag/search`   | POST   | RAG retrieval       |
| `/api/upload/pdf`   | POST   | Upload PDF file     |
 


## 🔄 ReAct Reflection Workflow

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Router    │ ◀── Intent Recognition
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Planner   │ ◀── Task Planning
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
              ┌────▶│  Executor   │ ◀── Tool Execution
              │     └──────┬──────┘
              │            ▼
              │     ┌─────────────┐
              │     │  Reflector  │ ◀── ReAct Reflection
              │     └──────┬──────┘
              │            │
              │     ┌──────┴──────┐
              │     │             │
              │ Refine Needed  Completed 
              │     │             │
              └─────┘             ▼
                          ┌─────────────┐
                          │   Critic    │ ◀── Generate Answer
                          └──────┬──────┘
                                 ▼
                          ┌─────────────┐
                          │    END      │
                          └─────────────┘
```

## 🛠️ Tech Stack

| Component            | Technology                 |
|----------------------|----------------------------|
| Backend Framework    | FastAPI                    |
| Agent Framework      | LangGraph                  |
| Large Language Model | qwen-max (Aliyun)          |
| Embedding Model      | text-embedding-v4 (Aliyun) |
| Database             | SQLite + SQLAlchemy        |
| File Parsing         | PyPDF2 / DocMind           |
| Web Search           | Bocha API                  |



## ⚠️ Notes

1. **API Key**: The DASHSCOPE_API_KEY environment variable must be set to use LLM features
2. **PDF Files**: Example uses CICC Annual Report.pdf, can be replaced with other reports
3. **Code Security**: Code interpreter runs in a sandbox environment; dangerous operations are prohibited



---


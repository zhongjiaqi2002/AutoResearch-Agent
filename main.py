"""
Financial Research Report Automated Analyst - FastAPI Main Entry
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.routers import router
from database.init_db import init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application Lifecycle Management"""
    # Initialize database on startup
    print("Initializing database...")
    init_database()

    # Create upload directory
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    yield

    # Cleanup on shutdown (if needed)
    print("Application shutting down")


# Create FastAPI application
app = FastAPI(
    title="Financial Research Report Automated Analyst",
    description="""
Multi-Agent Financial Analysis System Based on LangGraph

## Features

- **Intelligent Conversation**: Natural language interaction, understand user intentions
- **Text2SQL**: Natural language to SQL query conversion
- **Code Execution**: Python data analysis and visualization
- **Research Report Parsing**: PDF document parsing and RAG retrieval
- **Web Search**: Get latest market information
- **ReAct Reflection**: Multi-turn reasoning, self-improvement

## Technical Architecture

- LangGraph Multi-Agent Workflow
- qwen-max Large Language Model
- SQLite Database + Simulated Financial Data
- text-embedding-v4 Vector Embedding
""",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "finance-analyst"}


if __name__ == "__main__":
    import uvicorn

    print("""
╔══════════════════════════════════════════════════════════════╗
║           Financial Research Report Automated Analyst v1.0.0  ║
║                                                              ║
║   Multi-Agent Financial Analysis System Based on LangGraph   ║
║                                                              ║
║   Features:                                                  ║
║   - Text2SQL Data Query                                      ║
║   - Code Interpreter (Data Analysis/Visualization)           ║
║   - PDF Research Report Parsing                              ║
║   - Web Search                                               ║
║   - RAG Knowledge Base Retrieval                             ║
║   - ReAct Reflection Reasoning                               ║
║                                                              ║
║   API Docs: http://localhost:8000/docs                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
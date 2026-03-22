"""
FastAPI Route Definitions
"""
import os
import sys
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import FinanceAnalystAgent, analyze_query
from tools.text2sql import Text2SQLTool
from tools.code_executor import CodeExecutorTool
from tools.file_parser import PDFParserTool
from tools.web_searcher import WebSearchTool
from tools.rag_searcher import RAGSearchTool
from database.init_db import get_db_session, get_table_schema



router = APIRouter()

_agent = None
_rag_tool = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = FinanceAnalystAgent()
    return _agent


def get_rag_tool():
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = RAGSearchTool()
    return _rag_tool


class ChatRequest(BaseModel):
    query: str
    max_iterations: int = 3
    stream: bool = False


class SQLRequest(BaseModel):
    question: str


class CodeRequest(BaseModel):
    code: str
    data: Optional[Dict] = None


class SearchRequest(BaseModel):
    query: str
    freshness: str = "noLimit"
    count: int = 10


class RAGRequest(BaseModel):
    query: str
    top_k: int = 5


@router.get("/")
async def root():
    """Root route"""
    return {
        "name": "Automated Financial Research Analyst",
        "version": "1.0.0",
        "description": "Multi-agent financial analysis system based on LangGraph",
        "endpoints": {
            "/api/chat": "Main chat interface",
            "/api/sql/query": "Text2SQL query",
            "/api/code/execute": "Code execution",
            "/api/search": "Web search",
            "/api/rag/search": "RAG retrieval",
        }
    }


@router.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat interface - Intelligent financial analysis

    Supports:
    - Data query (Text2SQL)
    - In-depth analysis (code execution)
    - Research report retrieval (RAG)
    - Web search
    """
    agent = get_agent()

    if request.stream:
        async def generate():
            for event in agent.stream_analyze(request.query, request.max_iterations):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    else:
        result = agent.analyze(request.query, request.max_iterations)
        return result


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat interface"""
    agent = get_agent()

    async def generate():
        for event in agent.stream_analyze(request.query, request.max_iterations):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.post("/api/sql/query")
async def sql_query(request: SQLRequest):
    """Text2SQL query interface"""
    tool = Text2SQLTool()
    result = tool.run(request.question)
    return result


@router.post("/api/code/execute")
async def execute_code(request: CodeRequest):
    """Code execution interface"""
    tool = CodeExecutorTool()
    result = tool.run(request.code, request.data)
    return result


@router.post("/api/search")
async def web_search(request: SearchRequest):
    """Web search interface"""
    tool = WebSearchTool()
    result = tool.run(request.query, request.freshness, request.count)
    return result


@router.post("/api/rag/search")
async def rag_search(request: RAGRequest):
    """RAG retrieval interface"""
    rag = get_rag_tool()
    result = rag.run(request.query, request.top_k)
    return result


@router.post("/api/rag/load_pdf")
async def load_pdf_to_rag(file_path: str):
    """Load PDF into RAG knowledge base"""
    rag = get_rag_tool()
    result = rag.load_from_pdf(file_path)
    return result


@router.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and parse PDF"""
    upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    parser = PDFParserTool()
    result = parser.run(file_path)

    rag = get_rag_tool()
    rag_result = rag.load_from_pdf(file_path)

    return {
        "file_path": file_path,
        "parse_result": {
            "success": result.get("success"),
            "page_count": result.get("page_count", 0),
            "text_length": len(result.get("text_content", ""))
        },
        "rag_result": rag_result
    }



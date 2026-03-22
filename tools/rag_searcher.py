"""
RAG Retrieval Tool - Semantic Search based on Vector Database
"""
import os
import sys
import json
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding import get_embedding_service


class RAGSearchTool:
    """RAG Retrieval Tool Class"""

    name = "rag_search"
    description = """Retrieve relevant information from the research report knowledge base. Suitable for:
    - Query specific content in research reports
    - Search for analysis on specific topics
    - Obtain historical research report viewpoints"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.knowledge_base = []  # In-memory knowledge base
        self._initialized = False

    def _load_knowledge_base(self, documents: List[Dict[str, str]] = None):
        """
        Load knowledge base

        Args:
            documents: List of documents, each document contains 'content' and 'metadata'
        """
        if documents:
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                # Generate embedding vector
                embedding = self.embedding_service.embed_single(content)

                self.knowledge_base.append({
                    "content": content,
                    "metadata": metadata,
                    "embedding": embedding
                })

        self._initialized = True

    def add_document(self, content: str, metadata: Dict = None):
        """
        Add a single document to the knowledge base

        Args:
            content: Document content
            metadata: Metadata
        """
        embedding = self.embedding_service.embed_single(content)

        self.knowledge_base.append({
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding
        })

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of search results
        """
        if not self.knowledge_base:
            return []

        # Generate query embedding vector
        query_embedding = self.embedding_service.embed_single(query)

        # Calculate similarity
        results = []
        for doc in self.knowledge_base:
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": similarity
            })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute RAG retrieval

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            Retrieval results
        """
        results = self.search(query, top_k)

        return {
            "success": True,
            "query": query,
            "results": results,
            "total_count": len(results)
        }

    def load_from_pdf(self, pdf_path: str, chunk_size: int = 500):
        """
        Load knowledge base from PDF file

        Args:
            pdf_path: PDF file path
            chunk_size: Text chunk size
        """
        from tools.file_parser import PDFParserTool

        parser = PDFParserTool()
        result = parser.run(pdf_path)

        if result.get("success"):
            text = result.get("text_content", "")

            # Split by headers (### or ##)
            sections = self._split_by_headers(text)

            for i, section in enumerate(sections):
                if len(section.strip()) > 50:  # Filter short segments
                    self.add_document(
                        content=section.strip(),
                        metadata={
                            "source": pdf_path,
                            "section_index": i,
                            "type": "pdf"
                        }
                    )

            return {
                "success": True,
                "document_count": len(sections),
                "source": pdf_path
            }

        return {
            "success": False,
            "error": result.get("error", "PDF parsing failed")
        }

    def _split_by_headers(self, text: str) -> List[str]:
        """
        Split text by headers

        Split by ### first, then split by ## if no ### found
        """
        import re

        # First try splitting by ###
        sections = re.split(r'\n###\s+', text)

        if len(sections) <= 1:
            # If no ###, try splitting by ##
            sections = re.split(r'\n##\s+', text)

        if len(sections) <= 1:
            # If still no headers, split by paragraphs (consecutive newlines)
            sections = re.split(r'\n\n+', text)

        # Filter empty segments
        sections = [s for s in sections if s.strip()]

        return sections


# Tool function definition
RAG_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "Perform semantic search from the research report knowledge base to obtain relevant content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g., 'Company revenue growth', 'Risk factor analysis'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return, default 5",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


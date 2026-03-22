"""
PDF Parser
"""
import os
import sys
import json
import time
import base64
import httpx
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class PDFParserTool:
    """PDF Parser Tool Class - Using Aliyun DocMind"""

    name = "pdf_parser"
    description = """Analyze PDF research reports and extract text content, tables, and chart information.
    Applicable scenarios:
    -Read research report content
    -Extract financial statement data
    -Obtain key information from research reports"""

    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/docmind/document-analysis"

    def _submit_task(self, file_path: str) -> Optional[str]:
        """
        Submit PDF parsing task.

        Args:
            file_path: PDF path

        Returns:
            Task ID
        """
        # Read the file and encode in Base64
        with open(file_path, 'rb') as f:
            file_content = base64.standard_b64encode(f.read()).decode('utf-8')

        file_name = os.path.basename(file_path)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "docmind-v1",
            "input": {
                "file_type": "pdf",
                "file_name": file_name,
                "file_content": file_content
            },
            "parameters": {
                "parse_mode": "scan",  # suitable for complicated document
                "output_figure": True,
                "output_table": True
            }
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/async-submit",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("output", {}).get("task_id"):
                        return result["output"]["task_id"]
                    else:
                        print(f"Task Submission Failed: {result}")
                        return None
                else:
                    print(f"API request failed: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            print(f"Task submission exception: {e}")
            return None

    def _get_task_result(self, task_id: str, max_wait: int = 120) -> Optional[Dict]:
        """
        Get task result

        Args:
            task_id: Task ID
            max_wait: maximum waiting time

        Returns:
            Analyze result
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "task_id": task_id
        }

        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{self.base_url}/async-fetch",
                        headers=headers,
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        task_status = result.get("output", {}).get("task_status")

                        if task_status == "SUCCEEDED":
                            return result.get("output", {}).get("result")
                        elif task_status == "FAILED":
                            print(f"Task Failed: {result}")
                            return None
                        else:
                            # Task is still being processed
                            time.sleep(2)
                    else:
                        print(f"Failed to fetch result: {response.status_code}")
                        return None
            except Exception as e:
                print(f"Exception occurred while fetching result: {e}")
                time.sleep(2)

        print("Task Timeout")
        return None

    def _parse_result(self, result: Dict) -> Dict[str, Any]:
        """
        Parse the API response result

        Args:
            result: The raw result returned by the API

        Returns:
            Structured parsed result
        """
        parsed = {
            "text_content": "",
            "tables": [],
            "figures": [],
            "page_count": 0,
            "sections": []
        }

        if not result:
            return parsed

        # Extract the content
        pages = result.get("pages", [])
        parsed["page_count"] = len(pages)

        all_text = []
        for page in pages:
            page_text = []
            for block in page.get("blocks", []):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        page_text.append(text)
                elif block.get("type") == "table":
                    parsed["tables"].append({
                        "page": page.get("page_no", 0),
                        "content": block.get("table", {})
                    })
                elif block.get("type") == "figure":
                    parsed["figures"].append({
                        "page": page.get("page_no", 0),
                        "caption": block.get("caption", "")
                    })

            all_text.append("\n".join(page_text))

        parsed["text_content"] = "\n\n--- 分页 ---\n\n".join(all_text)

        return parsed

    def parse_local_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse local file

        Args:
            file_path: PDF path

        Returns:
            parse result
        """
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)

            result = {
                "success": True,
                "file_path": file_path,
                "page_count": len(reader.pages),
                "text_content": "",
                "pages": []
            }

            all_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                all_text.append(text)
                result["pages"].append({
                    "page_no": i + 1,
                    "text": text[:2000] if len(text) > 2000 else text
                })

            result["text_content"] = "\n\n".join(all_text)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def run(self, file_path: str, use_docmind: bool = True) -> Dict[str, Any]:
        """
        Parse PDF file

        Args:
            file_path: PDF path
            use_docmind: Whether to use the DocMind API (requires DASHSCOPE_API_KEY), default is True

        Returns:
            Parse result
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        if use_docmind and self.api_key:
            # Using DocMind API
            print(f"Parsing with DocMind API: {file_path}")
            task_id = self._submit_task(file_path)
            if task_id:
                result = self._get_task_result(task_id)
                if result:
                    parsed = self._parse_result(result)
                    parsed["success"] = True
                    parsed["method"] = "docmind"
                    return parsed
            print("DocMind parsing failed, falling back to local parsing")
        elif use_docmind and not self.api_key:
            print("DASHSCOPE_API_KEY not set, using local parsing")

        # Use local parsing（PyPDF2）
        result = self.parse_local_pdf(file_path)
        result["method"] = "local"
        return result


# Tool function definition（LangGraph）
PDF_PARSER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pdf_parser",
        "description": "Parse PDF research reports and extract text content. Suitable for reading research reports, "
                       "financial statements, and other PDF files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path to the PDF file"
                },
                "use_docmind": {
                    "type": "boolean",
                    "description": "Whether to use the DocMind API for advanced parsing, default is True (requires "
                                   "DASHSCOPE_API_KEY)",
                    "default": True
                }
            },
            "required": ["file_path"]
        }
    }
}


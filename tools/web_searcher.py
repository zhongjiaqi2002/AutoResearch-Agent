"""
Web Search Tool - Using Bocha API
"""
import os
import sys
import json
import httpx
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class WebSearchTool:
    """Web Search Tool Class - Using Bocha API"""

    name = "web_search"
    description = """Search the internet for the latest information. Suitable for:
    - Get the latest market trends and news
    - Query the latest company announcements
    - Search industry research reports
    - Obtain real-time financial information"""

    def __init__(self):
        self.api_key = settings.BOCHAAI_API_KEY
        self.base_url = settings.BOCHAAI_BASE_URL
        if not self.api_key:
            print("Warning: BOCHAAI_API_KEY is not set, please configure it in the .env file")

    def search(
            self,
            query: str,
            freshness: str = "noLimit",
            summary: bool = True,
            count: int = 10
    ) -> Dict[str, Any]:
        """
        Execute web search

        Args:
            query: Search keywords
            freshness: Time freshness filter (noLimit/day/week/month)
            summary: Whether to return AI summary
            count: Number of results to return

        Returns:
            Search results
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "freshness": freshness,
            "summary": summary,
            "count": count
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/web-search",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    return self._format_result(result, query)
                else:
                    return {
                        "success": False,
                        "error": f"API request failed: {response.status_code}",
                        "query": query
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    def _format_result(self, result: Dict, query: str) -> Dict[str, Any]:
        """
        Format search results

        Args:
            result: Raw API response
            query: Search query

        Returns:
            Formatted results
        """
        formatted = {
            "success": True,
            "query": query,
            "summary": "",
            "results": [],
            "total_count": 0
        }

        # Extract summary and web pages
        if "data" in result:
            data = result["data"]

            # Get AI summary
            if "summary" in data:
                formatted["summary"] = data["summary"]

            # Get search results
            if "webPages" in data and "value" in data["webPages"]:
                web_pages = data["webPages"]["value"]
                formatted["total_count"] = len(web_pages)

                for page in web_pages:
                    formatted["results"].append({
                        "title": page.get("name", ""),
                        "url": page.get("url", ""),
                        "snippet": page.get("snippet", ""),
                        "date": page.get("datePublished", ""),
                        "site_name": page.get("siteName", "")
                    })

        return formatted

    def run(
            self,
            query: str,
            freshness: str = "noLimit",
            count: int = 10
    ) -> Dict[str, Any]:
        """
        Execute search and return results

        Args:
            query: Search keywords
            freshness: Time freshness
            count: Number of results

        Returns:
            Search results
        """
        return self.search(query, freshness, True, count)


# Tool function definition
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for the latest information, including market trends, company "
                       "announcements, industry news, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, e.g., 'Kweichow Moutai latest financial report', 'New energy "
                                   "vehicle industry trend'"
                },
                "freshness": {
                    "type": "string",
                    "enum": ["noLimit", "day", "week", "month"],
                    "description": "Time freshness filter: noLimit(No limit), day(Within 1 day), week(Within 1 week), "
                                   "month(Within 1 month)",
                    "default": "noLimit"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return, default 10",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}



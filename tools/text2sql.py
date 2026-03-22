"""
Text2SQL Tool - Natural Language to SQL Query Converter
"""
import os
import sys
import json
import re
from typing import Dict, Any, List, Optional
from sqlalchemy import text

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm import get_llm_service
from database.init_db import get_db_session, get_table_schema


class Text2SQLTool:
    """Text2SQL Tool Class"""

    name = "text2sql"
    description = """Convert natural language questions to SQL queries and execute them. Suitable for:
    - Query basic stock information (market cap, industry, PE/PB, etc.)
    - Query financial data (revenue, profit, ROE, etc.)
    - Query market data (price, change rate, volume, etc.)
    - Query research report information (rating, target price, analyst opinions, etc.)
    - Perform data filtering and sorting (e.g., top 10 market cap, ROE > 15%, etc.)"""

    def __init__(self):
        self.llm = get_llm_service()
        self.schema = get_table_schema()

    def _generate_sql(self, question: str) -> str:
        """Generate SQL statement"""
        prompt = f"""You are a professional SQL generator. Generate the corresponding SQLite query based on the user's natural language question.

Database Schema:
{self.schema}

User Question: {question}

Requirements:
1. Return only the SQL statement, no explanations
2. Use standard SQLite syntax
3. Use accurate table and column names
4. Use JOIN for related queries if needed
5. Use ORDER BY and LIMIT appropriately to optimize results
6. Stock code is a string type, use quotes
7. Date format is 'YYYY-MM-DD'

SQL Statement:"""

        response = self.llm.simple_chat(prompt)

        # Extract SQL (remove markdown formatting)
        sql = response.strip()
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.strip()

        return sql

    def _execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute SQL statement"""
        session = get_db_session()
        try:
            result = session.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()

            # Convert to list of dictionaries
            data = [dict(zip(columns, row)) for row in rows]

            return {
                "success": True,
                "sql": sql,
                "data": data,
                "row_count": len(data)
            }
        except Exception as e:
            return {
                "success": False,
                "sql": sql,
                "error": str(e)
            }
        finally:
            session.close()

    def _format_result(self, question: str, result: Dict[str, Any]) -> str:
        """Format results into natural language"""
        if not result["success"]:
            return f"Query failed: {result['error']}\nGenerated SQL: {result['sql']}"

        if result["row_count"] == 0:
            return f"Query succeeded, but no matching data found.\nExecuted SQL: {result['sql']}"

        # Let LLM generate natural language response
        prompt = f"""Based on the SQL query results, answer the user's question in natural language.

User Question: {question}

Executed SQL: {result['sql']}

Query Results ({result['row_count']} total):
{json.dumps(result['data'][:20], ensure_ascii=False, indent=2)}

Please summarize the results clearly and professionally. Highlight key information if data volume is large."""

        response = self.llm.simple_chat(prompt)
        return response

    def run(self, question: str, return_raw: bool = False) -> Dict[str, Any]:
        """
        Execute Text2SQL

        Args:
            question: Natural language question
            return_raw: Whether to return raw data

        Returns:
            Result dictionary
        """
        # Generate SQL
        sql = self._generate_sql(question)

        # Execute SQL
        result = self._execute_sql(sql)

        # Format results
        if return_raw:
            return result

        answer = self._format_result(question, result)

        return {
            "question": question,
            "sql": sql,
            "raw_data": result.get("data", []),
            "row_count": result.get("row_count", 0),
            "answer": answer,
            "success": result["success"]
        }


# Tool function definition
TEXT2SQL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "text2sql",
        "description": "Convert natural language questions to SQL queries and execute them. Suitable for querying "
                       "stock information, financial data, market data, research reports, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question to query, e.g., 'Top 5 stocks by market cap', "
                                   "'Which stocks have ROE greater than 15%"
                }
            },
            "required": ["question"]
        }
    }
}



"""
LangGraph Node Definitions - Implementation of Each Agent Node
"""
import os
import sys
import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm import get_llm_service
from tools.text2sql import Text2SQLTool, TEXT2SQL_TOOL_SCHEMA
from tools.code_executor import CodeExecutorTool, CODE_EXECUTOR_TOOL_SCHEMA
from tools.file_parser import PDFParserTool, PDF_PARSER_TOOL_SCHEMA
from tools.web_searcher import WebSearchTool, WEB_SEARCH_TOOL_SCHEMA
from tools.rag_searcher import RAGSearchTool, RAG_SEARCH_TOOL_SCHEMA


class AgentState(TypedDict):
    """Agent State Definition"""
    messages: List[Dict[str, str]]
    query: str
    intent: str
    plan: List[str]
    current_step: int
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    reasoning_steps: List[str]
    reflections: List[str]
    final_answer: str
    should_continue: bool
    iteration: int
    max_iterations: int
    error: Optional[str]


class RouterNode:
    """Router Node - Identify user intent and route to appropriate processing flow"""

    def __init__(self):
        self.llm = get_llm_service()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Execute routing"""
        query = state["query"]

        prompt = f"""You are a router for a financial analysis assistant. Analyze the user's question and determine which processing method to use.

User Question: {query}

Available Intent Types:
1. data_query - Data Query: requires querying the database for specific data such as stocks, finances, market trends, etc.
2. analysis - In-depth Analysis: requires integrating multiple data sources for analysis, may require code execution and visualization
3. research - Research Report Retrieval: requires retrieving research report content or latest market information
4. general - General Q&A: simple financial knowledge Q&A, no data query required

Analyze the user intent and return JSON format:
{{
    "intent": "intent_type",
    "reason": "reason_for_judgment",
    "suggested_tools": ["list_of_suggested_tools"]
}}

Return only JSON, no other content."""

        response = self.llm.simple_chat(prompt)

        try:
            # Parse JSON
            result = json.loads(response.strip())
            intent = result.get("intent", "general")
            reason = result.get("reason", "")
            suggested_tools = result.get("suggested_tools", [])
        except json.JSONDecodeError:
            intent = "general"
            reason = "Failed to parse intent"
            suggested_tools = []

        return {
            "intent": intent,
            "reasoning_steps": state.get("reasoning_steps", []) + [
                f"[Router] Identified intent: {intent}，reason: {reason}"
            ],
            "suggested_tools:": suggested_tools
        }


class PlannerNode:
    """Planning Node - Create execution plan"""
    def __init__(self):
        self.llm = get_llm_service()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Create plan"""
        query = state["query"]
        intent = state["intent"]

        # For analysis tasks, code generation is required
        if intent == "analysis":
            return self._plan_with_code(query, state)

        prompt = f"""You are a financial analysis task planner. Based on the user's question and identified intent, create a detailed execution plan.

User Question: {query}
Identified Intent: {intent}

Available Tools:
1. text2sql - Convert natural language to SQL for database queries (stock info, financial data, market trends, research records)
2. code_executor - Execute Python code for data analysis and visualization
3. pdf_parser - Parse PDF research documents, available files: data/CICC Annual Report.pdf
4. web_search - Search for latest market information
5. rag_search - Retrieve relevant content from research report knowledge base

Create an execution plan and return JSON format:
{{
    "plan": [
        {{"step": 1, "action": "action_description", "tool": "tool_name", "params": {{"parameters"}}}},
        ...
    ],
    "reasoning": "planning_reason"
}}

Return only JSON, no other content."""

        response = self.llm.simple_chat(prompt)

        try:
            result = json.loads(response.strip())
            plan = result.get("plan", [])
            reasoning = result.get("reasoning", "")
        except json.JSONDecodeError:
            # Default plan
            if intent == "data_query":
                plan = [{"step": 1, "action": "Query data", "tool": "text2sql", "params": {"question": query}}]
            else:
                plan = [{"step": 1, "action": "Answer directly", "tool": None, "params": {}}]
            reasoning = "Using default plan"

        return {
            "plan": plan,
            "current_step": 0,
            "reasoning_steps": state.get("reasoning_steps", []) + [
                f"[Planner] Created {len(plan)} step plan: {reasoning}"
            ]
        }

    def _plan_with_code(self, query: str, state: AgentState) -> Dict[str, Any]:
        """Generate plan with Python code for analysis tasks"""

        # Step 1: Ask LLM what data is needed and generate analysis code
        prompt = f"""You are a financial data analysis expert. The user requires data analysis; create an analysis plan including code.

User Question: {query}

Available Data Tables:
1. stocks - Basic stock information (stock_code, stock_name, industry, sector, market_cap, pe_ratio, pb_ratio, listing_date)
2. financials - Financial data (stock_code, report_date, report_type, revenue, net_profit, roe, roa, total_assets, gross_margin, net_margin)
3. market_data - Market data (stock_code, trade_date, open_price, close_price, high_price, low_price, volume, amount, change_pct, turnover_rate)
4. research_reports - Research report information (stock_code, title, analyst, institution, rating, target_price, publish_date, summary)

Note: pe_ratio and pb_ratio are in the stocks table, not in financials

Generate analysis plan and return JSON format:
{{
    "data_query": "Description of data to query (for text2sql), must clearly specify which fields to return",
    "expected_fields": ["list_of_fields_to_be_returned_by_query"],
    "analysis_code": "Python analysis code (use pandas to process 'data' variable, matplotlib for plotting, save charts to output/charts/)",
    "reasoning": "analysis_logic_explanation"
}}

Important Rules:
1. data_query must explicitly specify all fields to query; if filtering an industry, do so at the SQL level (e.g., WHERE industry = 'Banking')
2. expected_fields lists actual field names returned by the query
3. analysis_code may only use fields listed in expected_fields, never other fields

Analysis Code Requirements:
1. Assume data has been fetched via text2sql and stored in variable 'data' (a list of dict)
2. Convert data to DataFrame using pd.DataFrame(data)
3. Process data with pandas, print key analysis results
4. For plotting, use matplotlib (plt.figure, plt.bar, plt.plot, etc.)
5. Important: Do NOT call plt.savefig(); system will capture charts automatically
6. Strictly use only fields listed in expected_fields, do not assume other fields exist
7. Available variables: pd, np, plt, datetime, data

For example, if the user asks "Top 5 banking stocks by market cap":
- data_query: "Query stocks in banking industry (industry='Banking'), return stock_code, stock_name, market_cap, sort descending by market cap, take top 5"
- expected_fields: ["stock_code", "stock_name", "market_cap"]
- analysis_code: use only stock_code, stock_name, market_cap

Return only JSON, no other content."""

        response = self.llm.simple_chat(prompt)

        try:
            result = json.loads(response.strip())
            data_query = result.get("data_query", query)
            analysis_code = result.get("analysis_code", "print('No analysis code')")
            reasoning = result.get("reasoning", "")
        except json.JSONDecodeError:
            # Default: simple query + basic analysis
            data_query = query
            analysis_code = """
import pandas as pd
df = pd.DataFrame(data)
print("Data Overview:")
print(df.describe())
print("\\nFirst 5 rows of data:")
print(df.head())
"""
            reasoning = "Using default analysis plan"

        # Build two-step plan: 1. Fetch data 2. Execute analysis code
        plan = [
            {
                "step": 1,
                "action": f"Query data: {data_query[:50]}...",
                "tool": "text2sql",
                "params": {"question": data_query}
            },
            {
                "step": 2,
                "action": "Execute data analysis code",
                "tool": "code_executor",
                "params": {"code": analysis_code, "use_previous_data": True}
            }
        ]

        return {
            "plan": plan,
            "current_step": 0,
            "reasoning_steps": state.get("reasoning_steps", []) + [
                f"[Planner] Analysis task, generated 2-step plan: {reasoning}"
            ]
        }


class ExecutorNode:
    """Execution Node - Execute specific tool calls"""

    def __init__(self):
        self.llm = get_llm_service()
        self.tools = {
            "text2sql": Text2SQLTool(),
            "code_executor": CodeExecutorTool(),
            "pdf_parser": PDFParserTool(),
            "web_search": WebSearchTool(),
            "rag_search": RAGSearchTool(),
        }

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Execute current step"""
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        tool_results = state.get("tool_results", [])
        reasoning_steps = state.get("reasoning_steps", [])

        if current_step >= len(plan):
            return {
                "should_continue": False,
                "reasoning_steps": reasoning_steps + ["[Executor] All steps completed"]
            }

        step = plan[current_step]
        tool_name = step.get("tool")
        params = step.get("params", {})
        action = step.get("action", "")

        reasoning_steps.append(f"[Executor] Step {current_step + 1}: {action}")

        if tool_name and tool_name in self.tools:
            tool = self.tools[tool_name]

            # # Adjust parameters based on tool type
            if tool_name == "text2sql":
                question = params.get("question", state["query"])
                result = tool.run(question)
            elif tool_name == "code_executor":
                code = params.get("code", "")
                data = params.get("data")

                # If use_previous_data is set, get data from previous text2sql result
                if params.get("use_previous_data") and tool_results:
                    # Find latest text2sql result
                    for prev_result in reversed(tool_results):
                        if prev_result.get("tool") == "text2sql":
                            prev_data = prev_result.get("result", {})
                            # text2sql returns data in raw_data field
                            raw_data = prev_data.get("raw_data") or prev_data.get("data")
                            if prev_data.get("success") and raw_data:
                                data = raw_data
                                reasoning_steps.append(f"[Executor] Retrieved {len(data)} records from text2sql")
                            else:
                                # text2sql failed or no data, use empty list
                                data = []
                                error_msg = prev_data.get("error", "No results from query")
                                reasoning_steps.append(f"[Warning] text2sql returned no valid data: {error_msg}")
                            break

                result = tool.run(code, data)
            elif tool_name == "pdf_parser":
                file_path = params.get("file_path", "data/CICC Annual Report.pdf")
                result = tool.run(file_path)
            elif tool_name == "web_search":
                query = params.get("query", state["query"])
                result = tool.run(query)
            elif tool_name == "rag_search":
                query = params.get("query", state["query"])
                result = tool.run(query)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            tool_results.append({
                "step": current_step + 1,
                "tool": tool_name,
                "params": params,
                "result": result
            })

            reasoning_steps.append(f"[Executor] Tool {tool_name} execution completed")
        else:
            # No tool needed, answer directly with LLM
            response = self.llm.simple_chat(
                f"Please answer this financial question: {state['query']}",
                "You are a professional financial analyst."
            )
            tool_results.append({
                "step": current_step + 1,
                "tool": "llm",
                "result": {"answer": response}
            })

        return {
            "current_step": current_step + 1,
            "tool_results": tool_results,
            "reasoning_steps": reasoning_steps,
            "should_continue": current_step + 1 < len(plan)
        }


class ReflectorNode:
    """Reflection Node - ReAct reflection mechanism"""

    def __init__(self):
        self.llm = get_llm_service()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Perform reflection"""
        query = state["query"]
        tool_results = state.get("tool_results", [])
        reflections = state.get("reflections", [])
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        # Build reflection prompt
        results_summary = json.dumps(tool_results, ensure_ascii=False, indent=2)[:3000]

        prompt = f"""You are a financial analysis reflection expert. Review the current analysis process and results for in-depth reflection.

User Question: {query}

Current Iteration: {iteration + 1}/{max_iterations}

Obtained Results:
{results_summary}

Please reflect and answer the following questions:
1. Does the current result fully answer the user's question?
2. Is additional data or analysis required?
3. Are there omissions or errors in the analysis process?
4. Is further in-depth investigation needed?

Return JSON format:
{{
    "reflection": "reflection_content",
    "is_complete": true/false,
    "missing_aspects": ["missing_aspects"],
    "suggested_actions": ["suggested_follow_up_actions"],
    "confidence": 0.0-1.0
}}

Return only JSON."""

        response = self.llm.simple_chat(prompt)

        try:
            result = json.loads(response.strip())
            reflection = result.get("reflection", "")
            is_complete = result.get("is_complete", True)
            confidence = result.get("confidence", 0.8)
            suggested_actions = result.get("suggested_actions", [])
        except json.JSONDecodeError:
            reflection = "Reflection parsing failed"
            is_complete = True
            confidence = 0.5
            suggested_actions = []

        reflections.append({
            "iteration": iteration + 1,
            "reflection": reflection,
            "confidence": confidence
        })

        # Decide whether to continue iteration
        should_continue = (
                not is_complete and
                iteration < max_iterations - 1 and
                confidence < 0.85 and
                len(suggested_actions) > 0
        )

        new_plan = []
        if should_continue and suggested_actions:
            # Generate new plan based on suggestions
            new_plan = self._generate_new_plan(suggested_actions, state)

        return {
            "reflections": reflections,
            "iteration": iteration + 1,
            "should_continue": should_continue,
            "plan": new_plan if new_plan else state.get("plan", []),
            "current_step": 0 if new_plan else state.get("current_step", 0),
            "reasoning_steps": state.get("reasoning_steps", []) + [
                f"[Reflector] Round {iteration + 1} reflection: {reflection[:100]}...",
                f"[Reflector] Confidence: {confidence:.2f}, Continue: {should_continue}"
            ]
        }

    def _generate_new_plan(self, suggested_actions: List[str], state: AgentState) -> List[Dict]:
        """Generate new plan based on suggestions"""
        new_plan = []
        for i, action in enumerate(suggested_actions[:3]):  # Max 3 new steps
            # Simple action-to-tool mapping
            tool = None
            if "data" in action or "query" in action:
                tool = "text2sql"
            elif "analysis" in action or "calculate" in action:
                tool = "code_executor"
            elif "search" in action or "latest" in action:
                tool = "web_search"
            elif "report" in action:
                tool = "rag_search"

            new_plan.append({
                "step": i + 1,
                "action": action,
                "tool": tool,
                "params": {"question": state["query"]} if tool == "text2sql" else {}
            })

        return new_plan


class CriticNode:
    """Evaluation Node - Generate final answer"""

    def __init__(self):
        self.llm = get_llm_service()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Generate final answer"""
        query = state["query"]
        tool_results = state.get("tool_results", [])
        reflections = state.get("reflections", [])
        reasoning_steps = state.get("reasoning_steps", [])

        # Organize all results
        results_summary = []
        for tr in tool_results:
            result = tr.get("result", {})
            if isinstance(result, dict):
                if "answer" in result:
                    results_summary.append(f"[{tr.get('tool', 'unknown')}]: {result['answer']}")
                elif "output" in result:
                    results_summary.append(f"[{tr.get('tool', 'unknown')}]: {result['output']}")
                elif "summary" in result:
                    results_summary.append(f"[{tr.get('tool', 'unknown')}]: {result['summary']}")

        prompt = f"""You are a professional financial analyst. Based on the collected information, provide a comprehensive, accurate, and professional response to the user.

User Question: {query}

Collected Information:
{chr(10).join(results_summary) if results_summary else 'No additional information'}

Reflection Records:
{json.dumps([r['reflection'] for r in reflections], ensure_ascii=False) if reflections else 'No reflections'}

Requirements:
1. Response must be professional and accurate
2. Specify data clearly if involved
3. Describe chart content if charts are present
4. Provide appropriate risk warnings
5. Language must be clear and easy to understand

Please provide the final answer:"""

        final_answer = self.llm.simple_chat(prompt)

        return {
            "final_answer": final_answer,
            "should_continue": False,
            "reasoning_steps": reasoning_steps + [
                "[Evaluator] Generated final answer",
                f"[Completed] Analysis finished, executed {len(tool_results)} tool calls, {len(reflections)} reflection rounds"
            ]
        }

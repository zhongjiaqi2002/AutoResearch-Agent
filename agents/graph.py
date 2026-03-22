"""
LangGraph Main Process - Financial Researcher Workflow
"""
import os
import sys
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes import RouterNode, PlannerNode, ExecutorNode, ReflectorNode, CriticNode


class FinanceAnalystState(TypedDict):
    """Financial Analyst Agent State"""
    # Input
    messages: List[Dict[str, str]]
    query: str

    # Routing Result
    intent: str

    # Planning Result
    plan: List[Dict[str, Any]]
    current_step: int

    # Execution Result
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # Reasoning Process
    reasoning_steps: List[str]

    # Reflection Result
    reflections: List[Dict[str, Any]]

    # Control Flow
    should_continue: bool
    iteration: int
    max_iterations: int

    # Output
    final_answer: str
    error: Optional[str]


def create_finance_analyst_graph():
    """
    Create Financial Analyst Workflow Graph

    Workflow:
    1. Router: Identify user intent
    2. Planner: Create execution plan
    3. Executor: Execute tool calls (loop)
    4. Reflector: ReAct reflection (may trigger replanning)
    5. Critic: Generate final answer

    Returns:
        Compiled StateGraph
    """
    # Create node instances
    router = RouterNode()
    planner = PlannerNode()
    executor = ExecutorNode()
    reflector = ReflectorNode()
    critic = CriticNode()

    # Create state graph
    workflow = StateGraph(FinanceAnalystState)

    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("reflector", reflector)
    workflow.add_node("critic", critic)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    workflow.add_edge("router", "planner")
    workflow.add_edge("planner", "executor")

    # Conditional edges for executor
    def should_continue_execution(state: FinanceAnalystState) -> Literal["executor", "reflector"]:
        """Determine whether to continue execution"""
        if state.get("should_continue", False):
            return "executor"
        return "reflector"

    workflow.add_conditional_edges(
        "executor",
        should_continue_execution,
        {
            "executor": "executor",
            "reflector": "reflector"
        }
    )

    # Conditional edges for reflector
    def should_reflect_again(state: FinanceAnalystState) -> Literal["planner", "critic"]:
        """Determine whether to replan"""
        if state.get("should_continue", False) and state.get("iteration", 0) < state.get("max_iterations", 3):
            return "planner"  # Replan
        return "critic"  # Generate answer

    workflow.add_conditional_edges(
        "reflector",
        should_reflect_again,
        {
            "planner": "planner",
            "critic": "critic"
        }
    )

    # End of evaluation node
    workflow.add_edge("critic", END)

    # Compile graph
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


class FinanceAnalystAgent:
    """Financial Analyst Agent Wrapper Class"""

    def __init__(self):
        self.graph = create_finance_analyst_graph()
        self.thread_id = 0

    def analyze(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Execute financial analysis

        Args:
            query: User query
            max_iterations: Maximum number of iterations

        Returns:
            Analysis result
        """
        self.thread_id += 1

        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "query": query,
            "intent": "",
            "plan": [],
            "current_step": 0,
            "tool_calls": [],
            "tool_results": [],
            "reasoning_steps": [],
            "reflections": [],
            "should_continue": True,
            "iteration": 0,
            "max_iterations": max_iterations,
            "final_answer": "",
            "error": None
        }

        config = {"configurable": {"thread_id": str(self.thread_id)}}

        try:
            # Execute workflow
            result = self.graph.invoke(initial_state, config)

            return {
                "success": True,
                "query": query,
                "intent": result.get("intent", ""),
                "answer": result.get("final_answer", ""),
                "reasoning_steps": result.get("reasoning_steps", []),
                "reflections": result.get("reflections", []),
                "tool_results": result.get("tool_results", []),
                "iterations": result.get("iteration", 0)
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "answer": f"Error occurred during analysis: {str(e)}"
            }

    def stream_analyze(self, query: str, max_iterations: int = 3):
        """
        Stream financial analysis execution

        Args:
            query: User query
            max_iterations: Maximum number of iterations

        Yields:
            Status updates for each step
        """
        self.thread_id += 1

        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "query": query,
            "intent": "",
            "plan": [],
            "current_step": 0,
            "tool_calls": [],
            "tool_results": [],
            "reasoning_steps": [],
            "reflections": [],
            "should_continue": True,
            "iteration": 0,
            "max_iterations": max_iterations,
            "final_answer": "",
            "error": None
        }

        config = {"configurable": {"thread_id": str(self.thread_id)}}

        try:
            for event in self.graph.stream(initial_state, config):
                for node_name, state in event.items():
                    yield {
                        "node": node_name,
                        "state": state,
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            yield {
                "node": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Convenience function
def analyze_query(query: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    Convenience analysis function

    Args:
        query: User query
        max_iterations: Maximum number of iterations

    Returns:
        Analysis result
    """
    agent = FinanceAnalystAgent()
    return agent.analyze(query, max_iterations)



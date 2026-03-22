"""
Code Executor Tool - Safely execute Python code for data analysis
"""
import os
import sys
import io
import base64
import traceback
from typing import Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure Chinese fonts
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CodeExecutorTool:
    """Python Code Executor"""

    name = "code_executor"
    description = """Execute Python code for data analysis and visualization. Suitable for:
    - Data calculation and statistical analysis
    - Generate charts and visualizations
    - Complex data processing logic
    - Financial indicator calculation"""

    def __init__(self, timeout: int = 30, max_output: int = 10000):
        self.timeout = timeout
        self.max_output = max_output

        # Ensure output directory exists (use relative path)
        self.output_dir = "output/charts"
        os.makedirs(self.output_dir, exist_ok=True)

        # Predefined safe imports
        self.safe_imports = {
            'pandas': 'pd',
            'numpy': 'np',
            'matplotlib.pyplot': 'plt',
            'json': 'json',
            'math': 'math',
            'datetime': 'datetime',
            'statistics': 'statistics',
        }

        # Forbidden modules
        self.forbidden_modules = {
            'os', 'subprocess', 'shutil', 'sys', 'socket',
            'requests', 'urllib', 'http', 'ftplib', 'smtplib',
            '__builtins__', 'eval', 'exec', 'compile', 'open'
        }

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe global variable environment"""
        import pandas as pd
        import numpy as np
        import json
        import math
        from datetime import datetime, timedelta
        import statistics

        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'reversed': reversed,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'format': format,
            },
            'pd': pd,
            'np': np,
            'plt': plt,
            'json': json,
            'math': math,
            'datetime': datetime,
            'timedelta': timedelta,
            'statistics': statistics,
            'OUTPUT_DIR': self.output_dir,  # Chart output directory
        }

        return safe_globals

    def _check_code_safety(self, code: str) -> Optional[str]:
        """Check code safety"""
        for forbidden in self.forbidden_modules:
            if f"import {forbidden}" in code or f"from {forbidden}" in code:
                return f"Forbidden module import: {forbidden}"
            if f"__{forbidden}__" in code:
                return f"Forbidden usage: __{forbidden}__"

        # 检查危险操作
        dangerous_patterns = [
            ('open(', 'Forbidden function: open()'),
            ('eval(', 'Forbidden function: eval()'),
            ('exec(', 'Forbidden function: exec()'),
            ('compile(', 'Forbidden function: compile()'),
            ('__import__', 'Forbidden function: __import__'),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in code:
                return message

        return None

    def _preprocess_code(self, code: str) -> str:
        """Preprocess code, remove import statements for already imported modules"""
        import re

        # Preloaded modules
        preloaded = {
            'pandas': 'pd',
            'numpy': 'np',
            'matplotlib.pyplot': 'plt',
            'matplotlib': None,
            'json': 'json',
            'math': 'math',
            'datetime': None,
            'statistics': 'statistics',
        }

        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            stripped = line.strip()
            skip = False

            # Check if it's an import statement
            if stripped.startswith('import ') or stripped.startswith('from '):
                for module in preloaded:
                    if f'import {module}' in stripped or f'from {module}' in stripped:
                        skip = True
                        break

            if not skip:
                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def run(self, code: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute Python code

        Args:
            code: Python code to execute
            data: Optional input data (injected as variable 'data')

        Returns:
            Execution result dictionary
        """
        # Safety check
        safety_error = self._check_code_safety(code)
        if safety_error:
            return {
                "success": False,
                "error": safety_error,
                "output": "",
                "figures": []
            }

        # Preprocess code, remove duplicate imports
        code = self._preprocess_code(code)

        # Create safe execution environment
        safe_globals = self._create_safe_globals()

        # Inject input data
        safe_globals['data'] = data if data else []

        # Inject input data
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Capture output streams
        plt.close('all')

        result = {
            "success": True,
            "output": "",
            "error": "",
            "figures": [],
            "variables": {}
        }

        # Core execution logic
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals)

            # Get console output
            result["output"] = stdout_capture.getvalue()[:self.max_output]
            result["error"] = stderr_capture.getvalue()

            # Extract generated figures
            figures = [plt.figure(num) for num in plt.get_fignums()]
            for i, fig in enumerate(figures):
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                result["figures"].append({
                    "index": i,
                    "base64": img_base64,
                    "format": "png"
                })
                buf.close()

            # Extract key result variables
            result_vars = ['result', 'output', 'answer', 'df', 'summary']
            for var in result_vars:
                if var in safe_globals:
                    val = safe_globals[var]
                    try:
                        if hasattr(val, 'to_dict'):
                            result["variables"][var] = val.to_dict()
                        elif hasattr(val, 'tolist'):
                            result["variables"][var] = val.tolist()
                        else:
                            result["variables"][var] = str(val)
                    except:
                        result["variables"][var] = str(val)

        except Exception as e:
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        finally:
            plt.close('all')

        return result


# Tool function definition
CODE_EXECUTOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "code_executor",
        "description": "Execute Python code for data analysis and visualization. Supports pandas, matplotlib, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute."
                },
                "data": {
                    "type": "object",
                    "description": "optional input data, injected as variable 'data' into the code."
                }
            },
            "required": ["code"]
        }
    }
}


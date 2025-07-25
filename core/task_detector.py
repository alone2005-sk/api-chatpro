"""
Task Detection and Classification System
Analyzes user input to determine the appropriate task type and parameters
"""

import re
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"
    SHELL_COMMAND = "shell_command"
    DATA_ANALYSIS = "data_analysis"
    TEXT_PROCESSING = "text_processing"
    GENERAL_QUERY = "general_query"

@dataclass
class TaskInfo:
    task_type: TaskType
    confidence: float
    parameters: Dict[str, Any]
    estimated_duration: int  # seconds
    language: str = None
    operation: str = None

class TaskDetector:
    def __init__(self):
        self.patterns = {
            TaskType.CODE_GENERATION: [
                r"write.*code|generate.*code|create.*function|implement.*algorithm",
                r"build.*app|develop.*program|code.*solution",
                r"python.*script|javascript.*function|html.*page"
            ],
            TaskType.FILE_OPERATION: [
                r"read.*file|open.*file|load.*file",
                r"write.*file|save.*file|create.*file",
                r"edit.*file|modify.*file|update.*file"
            ],
            TaskType.SHELL_COMMAND: [
                r"run.*command|execute.*command|terminal.*command",
                r"install.*package|npm.*install|pip.*install",
                r"git.*clone|git.*commit|docker.*run"
            ],
            TaskType.DATA_ANALYSIS: [
                r"analyze.*data|data.*analysis|statistical.*analysis",
                r"plot.*graph|create.*chart|visualize.*data",
                r"pandas.*dataframe|numpy.*array|matplotlib.*plot"
            ],
            TaskType.TEXT_PROCESSING: [
                r"summarize.*text|translate.*text|rewrite.*text",
                r"extract.*information|parse.*text|format.*text",
                r"grammar.*check|spell.*check|proofread"
            ]
        }
        
        self.language_patterns = {
            "python": r"python|\.py|pandas|numpy|django|flask",
            "javascript": r"javascript|js|node|react|vue|angular",
            "html": r"html|css|web.*page|website",
            "sql": r"sql|database|query|select.*from",
            "bash": r"bash|shell|terminal|command.*line",
            "java": r"java|\.java|spring|maven",
            "cpp": r"c\+\+|cpp|\.cpp|\.h",
            "go": r"golang|go.*lang|\.go",
            "rust": r"rust|\.rs|cargo"
        }
    
    async def analyze(self, prompt: str, context: Dict = None) -> TaskInfo:
        """Analyze prompt to detect task type and extract parameters"""
        prompt_lower = prompt.lower()
        
        # Detect task type
        task_type, confidence = self._detect_task_type(prompt_lower)
        
        # Extract parameters based on task type
        parameters = self._extract_parameters(prompt, task_type, context)
        
        # Detect programming language if relevant
        language = self._detect_language(prompt_lower) if task_type in [
            TaskType.CODE_GENERATION, TaskType.FILE_OPERATION
        ] else None
        
        # Estimate duration
        duration = self._estimate_duration(task_type, parameters)
        
        return TaskInfo(
            task_type=task_type,
            confidence=confidence,
            parameters=parameters,
            estimated_duration=duration,
            language=language
        )
    
    def _detect_task_type(self, prompt: str) -> tuple[TaskType, float]:
        """Detect the most likely task type"""
        scores = {}
        
        for task_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                score += matches
            scores[task_type] = score
        
        # If no specific patterns match, default to general query
        if not any(scores.values()):
            return TaskType.GENERAL_QUERY, 0.5
        
        # Return highest scoring task type
        best_task = max(scores, key=scores.get)
        confidence = min(scores[best_task] / 3.0, 1.0)  # Normalize confidence
        
        return best_task, confidence
    
    def _detect_language(self, prompt: str) -> str:
        """Detect programming language from prompt"""
        for language, pattern in self.language_patterns.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return language
        return "python"  # Default to Python
    
    def _extract_parameters(self, prompt: str, task_type: TaskType, context: Dict) -> Dict[str, Any]:
        """Extract task-specific parameters"""
        parameters = {}
        
        if task_type == TaskType.CODE_GENERATION:
            parameters.update({
                "complexity": self._assess_complexity(prompt),
                "requires_testing": "test" in prompt.lower(),
                "requires_documentation": "document" in prompt.lower() or "comment" in prompt.lower()
            })
        
        elif task_type == TaskType.FILE_OPERATION:
            # Extract file paths
            file_matches = re.findall(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']', prompt)
            if file_matches:
                parameters["file_path"] = file_matches[0]
            
            # Detect operation type
            if any(word in prompt.lower() for word in ["read", "open", "load"]):
                parameters["operation"] = "read"
            elif any(word in prompt.lower() for word in ["write", "save", "create"]):
                parameters["operation"] = "write"
            elif any(word in prompt.lower() for word in ["edit", "modify", "update"]):
                parameters["operation"] = "edit"
        
        elif task_type == TaskType.SHELL_COMMAND:
            # Extract command from prompt
            command_patterns = [
                r'run\s+["\']([^"\']+)["\']',
                r'execute\s+["\']([^"\']+)["\']',
                r'`([^`]+)`'
            ]
            for pattern in command_patterns:
                match = re.search(pattern, prompt)
                if match:
                    parameters["command"] = match.group(1)
                    break
        
        elif task_type == TaskType.DATA_ANALYSIS:
            parameters.update({
                "analysis_type": self._detect_analysis_type(prompt),
                "requires_visualization": any(word in prompt.lower() for word in ["plot", "chart", "graph", "visualize"])
            })
        
        elif task_type == TaskType.TEXT_PROCESSING:
            if "summarize" in prompt.lower():
                parameters["operation"] = "summarize"
            elif "translate" in prompt.lower():
                parameters["operation"] = "translate"
            elif "rewrite" in prompt.lower():
                parameters["operation"] = "rewrite"
            else:
                parameters["operation"] = "process"
        
        return parameters
    
    def _assess_complexity(self, prompt: str) -> str:
        """Assess code complexity based on prompt"""
        complexity_indicators = {
            "simple": ["simple", "basic", "easy", "quick"],
            "medium": ["moderate", "standard", "typical"],
            "complex": ["complex", "advanced", "sophisticated", "enterprise", "production"]
        }
        
        prompt_lower = prompt.lower()
        for level, indicators in complexity_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                return level
        
        # Default assessment based on length and technical terms
        technical_terms = len(re.findall(r'\b(class|function|algorithm|database|api|framework)\b', prompt_lower))
        if technical_terms > 3 or len(prompt) > 200:
            return "complex"
        elif technical_terms > 1 or len(prompt) > 100:
            return "medium"
        else:
            return "simple"
    
    def _detect_analysis_type(self, prompt: str) -> str:
        """Detect type of data analysis requested"""
        analysis_types = {
            "statistical": ["statistics", "statistical", "mean", "median", "correlation"],
            "visualization": ["plot", "chart", "graph", "visualize", "dashboard"],
            "machine_learning": ["ml", "machine learning", "predict", "model", "classification"],
            "exploratory": ["explore", "exploratory", "eda", "data exploration"]
        }
        
        prompt_lower = prompt.lower()
        for analysis_type, keywords in analysis_types.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return analysis_type
        
        return "general"
    
    def _estimate_duration(self, task_type: TaskType, parameters: Dict) -> int:
        """Estimate task duration in seconds"""
        base_durations = {
            TaskType.CODE_GENERATION: 30,
            TaskType.FILE_OPERATION: 5,
            TaskType.SHELL_COMMAND: 10,
            TaskType.DATA_ANALYSIS: 45,
            TaskType.TEXT_PROCESSING: 15,
            TaskType.GENERAL_QUERY: 10
        }
        
        base_duration = base_durations.get(task_type, 20)
        
        # Adjust based on complexity
        if task_type == TaskType.CODE_GENERATION:
            complexity = parameters.get("complexity", "simple")
            multipliers = {"simple": 1.0, "medium": 2.0, "complex": 4.0}
            base_duration *= multipliers.get(complexity, 1.0)
        
        return int(base_duration)

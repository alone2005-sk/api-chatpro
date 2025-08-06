"""
Advanced Task Detection and Intent Analysis
Automatically detects user intent and task types from natural language
"""

import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    PROJECT_CREATION = "project_creation"
    MOBILE_APP = "mobile_app"
    FILE_OPERATION = "file_operation"
    CODE_EXECUTION = "code_execution"
    BUG_FIXING = "bug_fixing"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    API_CREATION = "api_creation"
    DATABASE_DESIGN = "database_design"
    DEPLOYMENT = "deployment"
    ANALYSIS = "analysis"
    LEARNING = "learning"
    GENERAL_CHAT = "general_chat"

class ProjectPlatform(Enum):
    WEB = "web"
    MOBILE_ANDROID = "mobile_android"
    MOBILE_IOS = "mobile_ios"
    DESKTOP = "desktop"
    API = "api"
    ML_AI = "ml_ai"
    BLOCKCHAIN = "blockchain"
    GAME = "game"

@dataclass
class TaskInfo:
    task_type: TaskType
    confidence: float
    parameters: Dict[str, Any]
    estimated_duration: int
    complexity_level: int  # 1-5
    platform: Optional[ProjectPlatform] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    requires_testing: bool = False
    requires_deployment: bool = False

class TaskDetector:
    def __init__(self):
        self.task_patterns = {
            TaskType.CODE_GENERATION: [
                r"write.*code|generate.*code|create.*function|implement.*algorithm",
                r"build.*function|develop.*method|code.*solution",
                r"python.*script|javascript.*function|java.*class"
            ],
            TaskType.PROJECT_CREATION: [
                r"create.*project|build.*application|develop.*app",
                r"full.*stack|complete.*project|entire.*application",
                r"build.*from.*scratch|new.*project|start.*project"
            ],
            TaskType.MOBILE_APP: [
                r"mobile.*app|android.*app|ios.*app|react.*native",
                r"flutter.*app|kotlin.*app|swift.*app|mobile.*application",
                r"app.*store|play.*store|mobile.*development"
            ],
            TaskType.FILE_OPERATION: [
                r"read.*file|write.*file|edit.*file|delete.*file",
                r"file.*operation|manage.*files|handle.*files"
            ],
            TaskType.CODE_EXECUTION: [
                r"run.*code|execute.*code|test.*code|compile.*code",
                r"run.*script|execute.*program|test.*function"
            ],
            TaskType.BUG_FIXING: [
                r"fix.*bug|debug.*code|solve.*error|fix.*issue",
                r"error.*fixing|troubleshoot|resolve.*problem"
            ],
            TaskType.TESTING: [
                r"test.*code|unit.*test|integration.*test|write.*tests",
                r"testing.*framework|test.*suite|automated.*testing"
            ],
            TaskType.API_CREATION: [
                r"create.*api|build.*api|rest.*api|graphql.*api",
                r"api.*endpoint|web.*service|microservice"
            ],
            TaskType.DATABASE_DESIGN: [
                r"database.*design|create.*database|sql.*schema",
                r"data.*model|database.*structure|table.*design"
            ],
            TaskType.DEPLOYMENT: [
                r"deploy.*app|deployment|docker.*container|kubernetes",
                r"cloud.*deployment|aws.*deploy|heroku.*deploy"
            ]
        }
        
        self.language_patterns = {
            "python": r"python|\.py|django|flask|fastapi|pandas|numpy",
            "javascript": r"javascript|js|node|react|vue|angular|express",
            "typescript": r"typescript|ts|tsx|angular|nest",
            "java": r"java|\.java|spring|maven|gradle|android",
            "kotlin": r"kotlin|\.kt|android.*kotlin",
            "swift": r"swift|\.swift|ios|xcode",
            "dart": r"dart|flutter|\.dart",
            "go": r"golang|go.*lang|\.go",
            "rust": r"rust|\.rs|cargo",
            "php": r"php|laravel|symfony|wordpress",
            "ruby": r"ruby|rails|\.rb",
            "csharp": r"c#|csharp|\.net|asp\.net|unity",
            "cpp": r"c\+\+|cpp|\.cpp|\.h",
            "sql": r"sql|mysql|postgresql|sqlite|database"
        }
        
        self.framework_patterns = {
            "react": r"react|jsx|create-react-app",
            "vue": r"vue|vuejs|nuxt",
            "angular": r"angular|ng|typescript",
            "django": r"django|python.*web",
            "flask": r"flask|python.*api",
            "fastapi": r"fastapi|python.*api",
            "express": r"express|node.*js|npm",
            "spring": r"spring|java.*web|maven",
            "laravel": r"laravel|php.*framework",
            "rails": r"rails|ruby.*web",
            "flutter": r"flutter|dart|mobile",
            "react_native": r"react.*native|mobile.*react"
        }
        
        self.platform_patterns = {
            ProjectPlatform.WEB: r"web.*app|website|web.*application|frontend|backend",
            ProjectPlatform.MOBILE_ANDROID: r"android|kotlin|java.*mobile|play.*store",
            ProjectPlatform.MOBILE_IOS: r"ios|swift|xcode|app.*store|iphone",
            ProjectPlatform.DESKTOP: r"desktop|electron|tkinter|javafx|wpf",
            ProjectPlatform.API: r"api|rest|graphql|microservice|web.*service",
            ProjectPlatform.ML_AI: r"machine.*learning|ai|tensorflow|pytorch|ml",
            ProjectPlatform.BLOCKCHAIN: r"blockchain|crypto|smart.*contract|web3",
            ProjectPlatform.GAME: r"game|unity|unreal|pygame|game.*engine"
        }
    
    async def initialize(self):
        """Initialize the task detector"""
        logger.info("Task Detector initialized")
    
    async def analyze_intent(self, prompt: str, context: Dict = None) -> TaskInfo:
        """Analyze user prompt to detect intent and task type"""
        prompt_lower = prompt.lower()
        
        # Detect primary task type
        task_type, confidence = self._detect_task_type(prompt_lower)
        
        # Detect platform if applicable
        platform = self._detect_platform(prompt_lower)
        
        # Detect programming language
        language = self._detect_language(prompt_lower)
        
        # Detect framework
        framework = self._detect_framework(prompt_lower)
        
        # Extract parameters
        parameters = self._extract_parameters(prompt, task_type, context)
        
        # Assess complexity
        complexity = self._assess_complexity(prompt, task_type)
        
        # Estimate duration
        duration = self._estimate_duration(task_type, complexity, parameters)
        
        # Check if testing/deployment is needed
        requires_testing = self._requires_testing(prompt_lower, task_type)
        requires_deployment = self._requires_deployment(prompt_lower, task_type)
        
        return TaskInfo(
            task_type=task_type,
            confidence=confidence,
            parameters=parameters,
            estimated_duration=duration,
            complexity_level=complexity,
            platform=platform,
            language=language,
            framework=framework,
            requires_testing=requires_testing,
            requires_deployment=requires_deployment
        )
    
    def _detect_task_type(self, prompt: str) -> tuple[TaskType, float]:
        """Detect the most likely task type"""
        scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                score += matches
            scores[task_type] = score
        
        # Special logic for project creation
        if any(word in prompt for word in ["full", "complete", "entire", "whole"]):
            scores[TaskType.PROJECT_CREATION] += 2
        
        # Special logic for mobile apps
        if any(word in prompt for word in ["mobile", "android", "ios", "app"]):
            scores[TaskType.MOBILE_APP] += 2
        
        if not any(scores.values()):
            return TaskType.GENERAL_CHAT, 0.5
        
        best_task = max(scores, key=scores.get)
        confidence = min(scores[best_task] / 3.0, 1.0)
        
        return best_task, confidence
    
    def _detect_platform(self, prompt: str) -> Optional[ProjectPlatform]:
        """Detect target platform"""
        for platform, pattern in self.platform_patterns.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return platform
        return None
    
    def _detect_language(self, prompt: str) -> Optional[str]:
        """Detect programming language"""
        for language, pattern in self.language_patterns.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return language
        return None
    
    def _detect_framework(self, prompt: str) -> Optional[str]:
        """Detect framework"""
        for framework, pattern in self.framework_patterns.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return framework
        return None
    
    def _extract_parameters(self, prompt: str, task_type: TaskType, context: Dict) -> Dict[str, Any]:
        """Extract task-specific parameters"""
        parameters = {}
        
        if task_type == TaskType.PROJECT_CREATION:
            parameters.update({
                "include_tests": "test" in prompt.lower(),
                "include_docs": "document" in prompt.lower(),
                "include_ci_cd": any(word in prompt.lower() for word in ["ci", "cd", "deploy", "pipeline"]),
                "include_docker": "docker" in prompt.lower(),
                "database_needed": any(word in prompt.lower() for word in ["database", "db", "sql", "mongo"])
            })
        
        elif task_type == TaskType.MOBILE_APP:
            parameters.update({
                "cross_platform": any(word in prompt.lower() for word in ["flutter", "react native", "cross platform"]),
                "native": any(word in prompt.lower() for word in ["native", "kotlin", "swift"]),
                "ui_framework": self._detect_ui_framework(prompt.lower())
            })
        
        elif task_type == TaskType.API_CREATION:
            parameters.update({
                "api_type": "rest" if "rest" in prompt.lower() else "graphql" if "graphql" in prompt.lower() else "rest",
                "authentication": any(word in prompt.lower() for word in ["auth", "login", "jwt", "oauth"]),
                "database": any(word in prompt.lower() for word in ["database", "db", "sql"])
            })
        
        return parameters
    
    def _detect_ui_framework(self, prompt: str) -> Optional[str]:
        """Detect UI framework for mobile apps"""
        ui_frameworks = {
            "flutter": r"flutter|dart",
            "react_native": r"react.*native|rn",
            "native": r"native|kotlin|swift",
            "ionic": r"ionic|cordova",
            "xamarin": r"xamarin|c#"
        }
        
        for framework, pattern in ui_frameworks.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return framework
        
        return None
    
    def _assess_complexity(self, prompt: str, task_type: TaskType) -> int:
        """Assess project complexity (1-5)"""
        complexity_indicators = {
            1: ["simple", "basic", "easy", "quick", "minimal"],
            2: ["standard", "normal", "typical", "regular"],
            3: ["moderate", "medium", "intermediate", "full"],
            4: ["complex", "advanced", "comprehensive", "enterprise"],
            5: ["very complex", "sophisticated", "large scale", "production ready"]
        }
        
        prompt_lower = prompt.lower()
        
        # Check for explicit complexity indicators
        for level, indicators in complexity_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                return level
        
        # Assess based on task type and features
        base_complexity = {
            TaskType.CODE_GENERATION: 2,
            TaskType.PROJECT_CREATION: 4,
            TaskType.MOBILE_APP: 4,
            TaskType.API_CREATION: 3,
            TaskType.DATABASE_DESIGN: 3,
            TaskType.DEPLOYMENT: 3
        }.get(task_type, 2)
        
        # Adjust based on features mentioned
        feature_count = len(re.findall(r'\b(auth|database|api|test|deploy|docker|ci|cd)\b', prompt_lower))
        complexity_adjustment = min(feature_count // 2, 2)
        
        return min(base_complexity + complexity_adjustment, 5)
    
    def _estimate_duration(self, task_type: TaskType, complexity: int, parameters: Dict) -> int:
        """Estimate task duration in seconds"""
        base_durations = {
            TaskType.CODE_GENERATION: 60,
            TaskType.PROJECT_CREATION: 600,
            TaskType.MOBILE_APP: 900,
            TaskType.FILE_OPERATION: 10,
            TaskType.CODE_EXECUTION: 30,
            TaskType.BUG_FIXING: 120,
            TaskType.TESTING: 180,
            TaskType.API_CREATION: 300,
            TaskType.DATABASE_DESIGN: 240,
            TaskType.DEPLOYMENT: 180,
            TaskType.GENERAL_CHAT: 20
        }
        
        base_duration = base_durations.get(task_type, 60)
        complexity_multiplier = complexity * 0.5
        
        return int(base_duration * complexity_multiplier)
    
    def _requires_testing(self, prompt: str, task_type: TaskType) -> bool:
        """Check if task requires testing"""
        test_indicators = ["test", "testing", "unit test", "integration test", "automated test"]
        
        if any(indicator in prompt for indicator in test_indicators):
            return True
        
        # Automatically require testing for certain task types
        return task_type in [TaskType.PROJECT_CREATION, TaskType.API_CREATION, TaskType.MOBILE_APP]
    
    def _requires_deployment(self, prompt: str, task_type: TaskType) -> bool:
        """Check if task requires deployment"""
        deploy_indicators = ["deploy", "deployment", "production", "live", "host", "cloud"]
        
        if any(indicator in prompt for indicator in deploy_indicators):
            return True
        
        return task_type in [TaskType.PROJECT_CREATION, TaskType.API_CREATION]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get task detector status"""
        return {
            "supported_task_types": [task.value for task in TaskType],
            "supported_platforms": [platform.value for platform in ProjectPlatform],
            "supported_languages": list(self.language_patterns.keys()),
            "supported_frameworks": list(self.framework_patterns.keys()),
            "status": "ready"
        }
    async def detect_task(self, prompt: str, context: Dict = None) -> TaskInfo:
        """Alias for analyze_intent for backward compatibility"""
        return await self.analyze_intent(prompt, context)

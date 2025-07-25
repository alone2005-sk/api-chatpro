"""
AI Logger - Comprehensive logging and monitoring system
Logging configuration for DAMN BOT
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import sys

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(logs_dir / f"damn_bot_{timestamp}.log"))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

class AILogger:
    def __init__(self):
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        setup_logging(log_file=self.logs_dir / "ai_backend.log")
        
        self.logger = get_logger("AIBackend")
        self.task_logs = {}
    
    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        return f"task_{uuid.uuid4().hex[:12]}"
    
    def log_request(self, task_id: str, request: Any, task_info: Any):
        """Log incoming request"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "type": "request",
            "prompt": request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt,
            "task_type": task_info.task_type.value if hasattr(task_info.task_type, 'value') else str(task_info.task_type),
            "confidence": getattr(task_info, 'confidence', 0),
            "estimated_duration": getattr(task_info, 'estimated_duration', 0)
        }
        
        self._save_log(task_id, log_entry)
        self.logger.info(f"Request received - Task: {task_id}, Type: {task_info.task_type}")
    
    def log_completion(self, task_id: str, result: Dict[str, Any]):
        """Log task completion"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "type": "completion",
            "result_size": len(str(result)),
            "success": True
        }
        
        self._save_log(task_id, log_entry)
        self.logger.info(f"Task completed - {task_id}")
    
    def log_error(self, task_id: str, error: Exception):
        """Log task error"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "success": False
        }
        
        self._save_log(task_id, log_entry)
        self.logger.error(f"Task failed - {task_id}: {str(error)}")
    
    def log_llm_call(self, task_id: str, provider: str, prompt_length: int, response_length: int, duration: float):
        """Log LLM API call"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "type": "llm_call",
            "provider": provider,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "duration": duration
        }
        
        self._save_log(task_id, log_entry)
        self.logger.info(f"LLM call - Provider: {provider}, Duration: {duration:.2f}s")
    
    def _save_log(self, task_id: str, log_entry: Dict[str, Any]):
        """Save log entry to file"""
        if task_id not in self.task_logs:
            self.task_logs[task_id] = []
        
        self.task_logs[task_id].append(log_entry)
        
        # Save to file
        log_file = self.logs_dir / f"{task_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.task_logs[task_id], f, indent=2)
    
    def get_task_logs(self, task_id: str) -> List[Dict[str, Any]]:
        """Get logs for specific task"""
        log_file = self.logs_dir / f"{task_id}.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_tasks = len(list(self.logs_dir.glob("task_*.json")))
        
        # Count task types
        task_types = {}
        for log_file in self.logs_dir.glob("task_*.json"):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    for log in logs:
                        if log.get("type") == "request":
                            task_type = log.get("task_type", "unknown")
                            task_types[task_type] = task_types.get(task_type, 0) + 1
            except:
                continue
        
        return {
            "total_tasks": total_tasks,
            "task_types": task_types,
            "logs_directory": str(self.logs_dir)
        }

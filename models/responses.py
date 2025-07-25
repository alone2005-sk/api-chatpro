"""
Response Models - Pydantic models for API responses
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class TaskStatus(str, Enum):
    PROCESSING = "processing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    task_type: str = Field(..., description="Detected task type")
    estimated_duration: int = Field(..., description="Estimated completion time in seconds")
    message: str = Field(default="Task created successfully", description="Status message")

class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    created_at: str
    completed_at: Optional[str] = None
    duration: Optional[float] = None

class CodeGenerationResult(BaseModel):
    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language")
    provider: str = Field(..., description="LLM provider used")
    validation: Optional[Dict[str, Any]] = Field(default=None, description="Code validation results")
    file_path: Optional[str] = Field(default=None, description="Saved file path")

class FileOperationResult(BaseModel):
    operation: str = Field(..., description="Operation performed")
    file_path: str = Field(..., description="File path")
    success: bool = Field(..., description="Operation success status")
    size: Optional[int] = Field(default=None, description="File size in bytes")
    content: Optional[str] = Field(default=None, description="File content (for read operations)")

class ExecutionResult(BaseModel):
    status: str = Field(..., description="Execution status")
    exit_code: int = Field(..., description="Process exit code")
    output: str = Field(..., description="Standard output")
    error: str = Field(..., description="Error output")
    duration: Optional[float] = Field(default=None, description="Execution duration in seconds")

class SystemStatus(BaseModel):
    status: str = Field(..., description="System status")
    local_models: List[str] = Field(default_factory=list, description="Available local models")
    remote_apis: List[str] = Field(default_factory=list, description="Available remote APIs")
    active_tasks: int = Field(..., description="Number of active tasks")
    sandbox_status: Dict[str, Any] = Field(..., description="Sandbox environment status")

"""
Request Models - Pydantic models for API requests
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class TaskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000, description="User prompt/request")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    validate_code: bool = Field(default=True, description="Whether to validate generated code")
    save_output: bool = Field(default=True, description="Whether to save output to file")
    prefer_local: bool = Field(default=True, description="Prefer local LLMs over remote APIs")
    timeout: Optional[int] = Field(default=60, description="Task timeout in seconds")

class FileOperationRequest(BaseModel):
    operation: str = Field(..., description="Operation type: read, write, edit, delete")
    file_path: str = Field(..., description="File path")
    content: Optional[str] = Field(default=None, description="Content for write operations")
    encoding: str = Field(default="utf-8", description="File encoding")

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="Code to execute")
    language: str = Field(..., description="Programming language")
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    save_output: bool = Field(default=False, description="Save execution output")

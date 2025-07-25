"""
Request and response models for DAMN BOT
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from fastapi import UploadFile

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

class ChatRequest(BaseModel):
    """Chat request model"""
    prompt: str = Field(..., description="User prompt")
    files: List[UploadFile] = Field(default=[], description="Uploaded files")
    web_search: bool = Field(default=False, description="Enable web search")
    voice: bool = Field(default=False, description="Generate voice output")
    project_id: Optional[str] = Field(default=None, description="Project session ID")
    stream: bool = Field(default=False, description="Enable streaming response")
    research_mode: bool = Field(default=False, description="Enable deep research mode")
    deep_learning: bool = Field(default=False, description="Enable deep learning features")
    code_execution: bool = Field(default=True, description="Enable code execution")
    auto_fix: bool = Field(default=True, description="Auto-fix code errors")
    language: str = Field(default="auto", description="Programming language preference")
    max_iterations: int = Field(default=3, description="Max iterations for auto-fix")
    
    class Config:
        arbitrary_types_allowed = True

class ChatResponse(BaseModel):
    """Chat response model"""
    project_id: str = Field(..., description="Project session ID")
    message_id: str = Field(..., description="Message ID")
    text: str = Field(..., description="Response text")
    code: Optional[str] = Field(default=None, description="Generated code")
    language: Optional[str] = Field(default=None, description="Code language")
    files: List[str] = Field(default=[], description="Generated file paths")
    audio_file: Optional[str] = Field(default=None, description="Generated audio file path")
    sources: List[Dict[str, Any]] = Field(default=[], description="Web search sources")
    research_data: Optional[Dict[str, Any]] = Field(default=None, description="Research results")
    execution_results: Optional[Dict[str, Any]] = Field(default=None, description="Code execution results")
    llm_scores: Dict[str, float] = Field(default={}, description="LLM performance scores")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

class FileAnalysis(BaseModel):
    """File analysis result"""
    filename: str
    file_type: str
    size: int
    content: str
    metadata: Dict[str, Any] = Field(default={})
    analysis_time: float

class WebSearchResult(BaseModel):
    """Web search result"""
    query: str
    sources: List[Dict[str, Any]]
    summary: str
    timestamp: str

class VoiceGeneration(BaseModel):
    """Voice generation result"""
    text: str
    audio_file: str
    engine: str
    duration: float
    metadata: Dict[str, Any] = Field(default={})

class CodeExecution(BaseModel):
    """Code execution result"""
    code: str
    language: str
    output: str
    error: Optional[str] = None
    exit_code: int
    execution_time: float
    success: bool
    iterations: int = 1

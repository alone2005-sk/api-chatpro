"""
Request and response models for DAMN BOT AI System
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
    prompt: str = Field(..., description="User prompt/question")
    files: List[UploadFile] = Field(default=[], description="Uploaded files")
    project_id: Optional[str] = Field(None, description="Project ID for conversation continuity")
    web_search: bool = Field(False, description="Enable web search")
    voice: bool = Field(False, description="Generate voice response")
    research_mode: bool = Field(False, description="Enable deep research mode")
    deep_learning: bool = Field(False, description="Enable deep learning processing")
    code_execution: bool = Field(False, description="Enable code execution")
    auto_fix: bool = Field(False, description="Auto-fix code errors")
    max_iterations: int = Field(3, description="Maximum auto-fix iterations")
    
    class Config:
        arbitrary_types_allowed = True

class ChatResponse(BaseModel):
    """Chat response model"""
    project_id: str = Field(..., description="Project ID")
    message_id: str = Field(..., description="Message ID")
    text: str = Field(..., description="AI response text")
    code: Optional[str] = Field(None, description="Generated code")
    language: Optional[str] = Field(None, description="Programming language")
    files: List[Dict[str, Any]] = Field(default=[], description="Generated files")
    audio_file: Optional[str] = Field(None, description="Generated audio file path")
    sources: List[Dict[str, Any]] = Field(default=[], description="Web search sources")
    research_data: Optional[Dict[str, Any]] = Field(None, description="Research findings")
    execution_results: Optional[Dict[str, Any]] = Field(None, description="Code execution results")
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
    ocr_results: Optional[Dict[str, Any]] = Field(default=None)
    object_detection: Optional[Dict[str, Any]] = Field(default=None)
    transcription: Optional[Dict[str, Any]] = Field(default=None)

class FileProcessResult(BaseModel):
    """File processing result"""
    filename: str
    file_type: str
    size: int
    content: str
    metadata: Dict[str, Any] = {}
    processing_time: float
    success: bool
    error: Optional[str] = None

class CodeExecutionResult(BaseModel):
    """Code execution result"""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float
    language: str
    code: str

class WebSearchResult(BaseModel):
    """Web search result"""
    query: str
    results: List[Dict[str, Any]]
    summary: str
    sources: List[Dict[str, Any]]
    search_time: float

class VoiceGenerationResult(BaseModel):
    """Voice generation result"""
    audio_file: str
    duration: float
    voice_model: str
    text: str
    generation_time: float

"""
Request and response models for DAMN BOT AI System
"""
from datetime import datetime

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from fastapi import UploadFile
class ChatMessage(BaseModel):

    
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    message_type: str = Field(default="text", description="Message type")
    timestamp: Optional[str] = None

class UnifiedChatRequest(BaseModel):
    """Unified request model for all chat capabilities"""
    prompt: str = Field(..., min_length=1, max_length=5000, description="User message")
    files: List[UploadFile] = Field([], description="Uploaded files")
    chat_id: Optional[str] = Field(None, description="Existing chat session ID")
    user_id: str = Field(default="anonymous", description="User identifier")
    stream: bool = Field(False, description="Enable streaming response")
    
    # AI Options
    enable_research: bool = Field(False, description="Enable deep research")
    research_depth: str = Field("comprehensive", description="Research depth")
    enable_learning: bool = Field(True, description="Enable self-learning")
    prefer_local: bool = Field(True, description="Prefer local LLMs")
    
    # Generation Options
    auto_detect_project: bool = Field(True, description="Auto-detect project creation")
    auto_detect_media: bool = Field(True, description="Auto-detect media generation")
    
    # Advanced Features
    web_search: bool = Field(False, description="Enable web search")
    voice: bool = Field(False, description="Enable voice generation")
    research_mode: bool = Field(False, description="Enable research mode")
    deep_learning: bool = Field(False, description="Enable deep learning processing")
    code_execution: bool = Field(True, description="Enable code execution")
    auto_fix: bool = Field(True, description="Enable auto-fixing of code")
    language: str = Field("auto", description="Programming language for code tasks")
    max_iterations: int = Field(3, description="Max iterations for code fixing")
    
    # Context
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class EnhancedChatResponse(BaseModel):
    """Enhanced response model with all AI capabilities"""
    success: bool = Field(..., description="Request success status")
    chat_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message ID")
    
    # Response content
    response: str = Field(..., description="AI response text")
    response_type: str = Field("text", description="Response type")
    
    # Generated content
    project_id: Optional[str] = Field(None, description="Generated project ID")
    media_files: List[Dict[str, Any]] = Field([], description="Generated media files")
    research_session_id: Optional[str] = Field(None, description="Research session ID")
    code: Optional[str] = Field(None, description="Generated code content")
    language: Optional[str] = Field(None, description="Programming language of generated code")
    audio_file: Optional[str] = Field(None, description="URL to generated audio file")
    research_data: Optional[Dict] = Field(None, description="Research findings")
    execution_results: Optional[Dict] = Field(None, description="Code execution results")
    
    # Metadata
    model_used: Optional[str] = Field(None, description="LLM model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    tokens_used: int = Field(0, description="Tokens consumed")
    confidence_score: float = Field(0.0, description="Response confidence")
    llm_scores: Dict[str, float] = Field(default_factory=dict, description="LLM quality scores")
    
    # Learning
    learning_applied: bool = Field(False, description="Self-learning was applied")
    patterns_used: int = Field(0, description="Learning patterns used")
    
    # Sources
    sources: List[Dict[str, Any]] = Field([], description="Research sources")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ProjectStatusResponse(BaseModel):
    project_id: str
    name: str
    status: str
    progress: int
    created_at: str
    completed_at: Optional[str] = None
    file_path: Optional[str] = None
    zip_path: Optional[str] = None
    metadata: Dict[str, Any] = {}

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

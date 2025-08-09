"""
DAMN BOT - Professional All-in-One AI Chat Agent
FastAPI backend with multi-LLM orchestration, file processing, web search, and voice generation
"""

import os

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from core.config import Settings
from core.database import DatabaseManager, init_database
from core.logger import setup_logging, get_logger
from services.chat_service import ChatService
from services.llm_orchestrator import LLMOrchestrator
from services.file_processor import FileProcessor
from services.web_search import WebSearchService
from services.voice_service import VoiceService
from services.code_service import CodeService
from services.research_service import ResearchService
from services.deep_learning_service import DeepLearningService
from models.requests import ChatRequest, ChatResponse
from middleware.security import SecurityMiddleware
from middleware.rate_limiter import RateLimitMiddleware
from services.research_service import ResearchService
from services.deep_research import DeepResearchService
from services.self_learning import SelfLearningService
from services.project_generator import ProjectGenerationService
from services.image_video_generator import ImageVideoGenerator
from models.requests import UnifiedChatRequest, EnhancedChatResponse, ProjectStatusResponse
# Initialize settings and logging
settings = Settings()
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

# Global service instances
chat_service = None
llm_orchestrator = None
file_processor = None
web_search_service = None
voice_service = None
code_service = None
research_service = None
deep_learning_service = None
db_manager = None
deep_research_service = None
self_learning_service = None
project_generation_service = None
image_video_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global chat_service, llm_orchestrator, file_processor, web_search_service
    global voice_service, code_service, research_service, deep_learning_service, db_manager
    global db_manager, llm_orchestrator, deep_research_service
    global self_learning_service, project_generation_service, image_video_generator
    
    logger.info("🚀 Starting DAMN BOT AI Chat Agent...")
    
    try:
        # Initialize database
        db_manager = DatabaseManager(settings.DATABASE_URL)
        await init_database(db_manager)
        
        # Initialize all services
        llm_orchestrator = LLMOrchestrator(settings)
        await llm_orchestrator.initialize()
        
        file_processor = FileProcessor(settings)
        await file_processor.initialize()
        
        web_search_service = WebSearchService(settings)
        await web_search_service.initialize()
        
        voice_service = VoiceService(settings)
        await voice_service.initialize()
        # deep_research_service = DeepResearchService(llm_orchestrator, db_manager)
        # await deep_research_service.initialize()
        
        self_learning_service = SelfLearningService(llm_orchestrator, db_manager)
        await self_learning_service.initialize()
        
        project_generation_service = ProjectGenerationService(llm_orchestrator, db_manager)
        await project_generation_service.initialize()
        
        image_video_generator = ImageVideoGenerator(llm_orchestrator)
        await image_video_generator.initialize()
        
        # code_service = CodeService(settings)
        # await code_service.initialize()
        
        research_service = ResearchService(
    settings=settings,
    web_search_service=web_search_service,
    llm_orchestrator=llm_orchestrator
          )
        #await research_service.initialize()
        
        # deep_learning_service = DeepLearningService(settings)
        # await deep_learning_service.initialize()
        
        chat_service = ChatService(
    db_manager=db_manager,
    llm_orchestrator=llm_orchestrator,
    file_processor=file_processor,
    web_search_service=web_search_service,
    voice_service=voice_service,
    code_service=code_service,
    research_service=research_service,
    deep_learning_service=deep_learning_service,
    project_generation_service=project_generation_service,
    self_learning_service=self_learning_service,
    image_video_generator=image_video_generator,
    settings=settings
)

        logger.info("✅ DAMN BOT AI Chat Agent is ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize DAMN BOT: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down DAMN BOT...")
    if llm_orchestrator:
        await llm_orchestrator.cleanup()
    if voice_service:
        await voice_service.cleanup()
    if code_service:
        await code_service.cleanup()
    if deep_learning_service:
        await deep_learning_service.cleanup()
    if db_manager:
        await db_manager.close()
    logger.info("✅ DAMN BOT shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="DAMN BOT - AI Chat Agent",
    description="Professional All-in-One AI Chat Agent with Multi-LLM Orchestration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_CALLS, window_seconds=settings.RATE_LIMIT_PERIOD)


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if os.path.exists(settings.GENERATED_MEDIA_DIR):
    app.mount("/media", StaticFiles(directory=settings.GENERATED_MEDIA_DIR), name="media")

# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    message_type: str = Field(default="text", description="Message type")
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    chat_id: Optional[str] = Field(None, description="Chat session ID")
    user_id: str = Field(default="anonymous", description="User identifier")
    
    # AI Options
    enable_research: bool = Field(default=False, description="Enable deep research")
    research_depth: str = Field(default="comprehensive", description="Research depth")
    enable_learning: bool = Field(default=True, description="Enable self-learning")
    prefer_local: bool = Field(default=True, description="Prefer local LLMs")
    
    # Generation Options
    auto_detect_project: bool = Field(default=True, description="Auto-detect project creation")
    auto_detect_media: bool = Field(default=True, description="Auto-detect media generation")
    
    # Context
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")

class ChatResponse(BaseModel):
    success: bool = Field(..., description="Request success status")
    chat_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message ID")
    
    # Response content
    response: str = Field(..., description="AI response text")
    response_type: str = Field(default="text", description="Response type")
    
    # Generated content
    project_id: Optional[str] = Field(None, description="Generated project ID")
    media_files: List[Dict[str, Any]] = Field(default=[], description="Generated media files")
    research_session_id: Optional[str] = Field(None, description="Research session ID")
    
    # Metadata
    model_used: Optional[str] = Field(None, description="LLM model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    tokens_used: int = Field(default=0, description="Tokens consumed")
    confidence_score: float = Field(default=0.0, description="Response confidence")
    
    # Learning
    learning_applied: bool = Field(default=False, description="Self-learning was applied")
    patterns_used: int = Field(default=0, description="Learning patterns used")
    
    # Sources
    research_sources: List[Dict[str, Any]] = Field(default=[], description="Research sources")
    
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

# Main chat endpoint
@app.post("/chat", response_model=EnhancedChatResponse)
async def unified_chat_endpoint(
    background_tasks: BackgroundTasks,
    prompt: Optional[str] = Form(None),
    files: List[UploadFile] = File(None),
    web_search: bool = Form(False),
    voice: bool = Form(False),
    chat_id: Optional[str] = Form(None),
    user_id: str = Form("anonymous"),
    stream: bool = Form(False),
    research_mode: bool = Form(False),
    deep_learning: bool = Form(False),
    code_execution: bool = Form(True),
    auto_fix: bool = Form(True),
    language: Optional[str] = Form("auto"),
    max_iterations: int = Form(3),
    enable_research: bool = Form(False),
    research_depth: str = Form("comprehensive"),
    enable_learning: bool = Form(True),
    prefer_local: bool = Form(True),
    auto_detect_project: bool = Form(True),
    auto_detect_media: bool = Form(True),
    context: Optional[str] = Form("{}")
):
    """Unified AI chat endpoint with comprehensive capabilities"""
    try:
        request = UnifiedChatRequest(
            prompt=prompt or "",
            files=files or [],
            web_search=web_search,
            voice=voice,
            chat_id=chat_id,
            user_id=user_id,
            stream=stream,
            research_mode=research_mode,
            deep_learning=deep_learning,
            code_execution=code_execution,
            auto_fix=auto_fix,
            language=language,
            max_iterations=max_iterations,
            enable_research=enable_research,
            research_depth=research_depth,
            enable_learning=enable_learning,
            prefer_local=prefer_local,
            auto_detect_project=auto_detect_project,
            auto_detect_media=auto_detect_media,
            context=json.loads(context) if context else {}
        )
        
        if request.stream:
            return StreamingResponse(
                chat_service.process_chat_stream(request),
                media_type="text/event-stream"
            )
            
        return await chat_service.process_chat(request, background_tasks)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{user_id}")
async def get_user_chats(user_id: str, limit: int = 50):
    """Get user's chat sessions"""
    try:
        return await chat_service.get_user_chats(user_id, limit)
    except Exception as e:
        logger.error(f"Get user chats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str, limit: int = 50, offset: int = 0):
    """Get chat messages with pagination"""
    try:
        return await chat_service.get_chat_messages(chat_id, limit, offset)
    except Exception as e:
        logger.error(f"Get chat messages error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session"""
    try:
        return await chat_service.delete_chat(chat_id)
    except Exception as e:
        logger.error(f"Delete chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/status", response_model=ProjectStatusResponse)
async def get_project_status(project_id: str):
    """Get project generation status"""
    try:
        return await chat_service.get_project_status(project_id)
    except Exception as e:
        logger.error(f"Get project status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/download")
async def download_project(project_id: str):
    """Download generated project as ZIP"""
    try:
        zip_path = await chat_service.create_project_zip(project_id)
        return FileResponse(
            path=zip_path,
            filename=f"project_{project_id}.zip",
            media_type="application/zip"
        )
    except Exception as e:
        logger.error(f"Download project error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Media generation endpoints
@app.post("/generate/image")
async def generate_image(
    prompt: str = Form(...),
    style: str = Form("photorealistic"),
    size: str = Form("1024x1024"),
    user_id: str = Form("anonymous")
):
    """Generate image from text prompt"""
    try:
        result = await image_video_generator.generate_image(
            prompt=prompt,
            style=style,
            size=size,
            user_id=user_id
        )
        
        if result.get("success"):
            return {
                "success": True,
                "image_url": f"/media/{os.path.basename(result['image_path'])}",
                "image_path": result["image_path"],
                "provider_used": result.get("provider_used"),
                "generation_time": result.get("generation_time", 0),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Image generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate/video")
async def generate_video(
    prompt: str = Form(...),
    style: str = Form("cinematic"),
    duration: int = Form(5),
    user_id: str = Form("anonymous")
):
    try:
        result = await image_video_generator.generate_video(
            prompt=prompt,
            style=style,
            duration=duration,
            user_id=user_id
        )
        if result.get("success"):
            return {
                "success": True,
                "video_url": f"/media/{os.path.basename(result['video_path'])}",
                "video_path": result["video_path"],
                "provider_used": result.get("provider_used"),
                "generation_time": result.get("generation_time", 0),
                "timestamp": datetime.now().isoformat()
            }
        else:
            error_msg = result.get("error", "Video generation failed")
            logger.error(f"Video generation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        logger.exception("Exception during video generation")  # This logs the full stack trace
        raise HTTPException(status_code=500, detail=str(e))

# Research endpoints
@app.post("/research")
async def conduct_research(
    query: str = Form(...),
    research_type: str = Form("comprehensive"),
    chat_id: Optional[str] = Form(None),
    user_id: str = Form("anonymous")
):
    """Conduct deep research on a topic"""
    try:
        # Create temporary chat if none provided
        if not chat_id:
            chat = await db_manager.create_chat(
                user_id=user_id,
                title=f"Research: {query[:50]}",
                description="Research session"
            )
            chat_id = chat.id
        
        result = await deep_research_service.conduct_research(
            query=query,
            chat_id=chat_id,
            research_type=research_type
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "session_id": result["session_id"],
            "query": result["query"],
            "research_type": result["research_type"],
            "report": result["report"],
            "sources_count": len(result.get("sources", [])),
            "processing_time": result["processing_time"],
            "confidence_score": result["report"].get("confidence_score", 0.0),
            "timestamp": result["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Learning endpoints
@app.post("/feedback")
async def submit_feedback(
    message_id: str = Form(...),
    rating: int = Form(..., ge=1, le=5),
    helpful: bool = Form(False),
    accurate: bool = Form(False),
    comments: Optional[str] = Form(None)
):
    """Submit feedback for AI response"""
    try:
        # This would update the learning system with user feedback
        # For now, we'll just acknowledge the feedback
        
        feedback_data = {
            "rating": rating,
            "helpful": helpful,
            "accurate": accurate,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
        # In a full implementation, this would update the message in the database
        # and trigger learning updates
        
        return {
            "success": True,
            "message": "Feedback received successfully",
            "feedback_id": f"feedback_{int(time.time())}"
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# System status endpoints
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {}
        }
        
        # Check each service
        if llm_orchestrator:
            health_status["services"]["llm_orchestrator"] = await llm_orchestrator.health_check()
        
        if deep_research_service:
            health_status["services"]["deep_research"] = await deep_research_service.health_check()
        
        if self_learning_service:
            health_status["services"]["self_learning"] = await self_learning_service.health_check()
        
        if project_generation_service:
            health_status["services"]["project_generation"] = await project_generation_service.health_check()
        
        if image_video_generator:
            health_status["services"]["media_generation"] = await image_video_generator.health_check()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time(),  # This would be calculated properly
            "services": {}
        }
        
        # Get stats from each service
        if llm_orchestrator:
            stats["services"]["llm_orchestrator"] = await llm_orchestrator.get_provider_stats()
        
        if deep_research_service:
            stats["services"]["deep_research"] = await deep_research_service.get_research_stats()
        
        if self_learning_service:
            stats["services"]["self_learning"] = await self_learning_service.get_learning_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Professional AI Agent Backend",
        "version": "1.0.0",
        "description": "Complete AI backend with chat history, project generation, media creation, and self-learning",
        "features": [
            "Chat with history management",
            "Automatic project detection and generation",
            "Image and video generation",
            "Deep research with multiple sources",
            "Self-learning from user interactions",
            "Professional code generation",
            "Real-time status updates"
        ],
        "endpoints": {
            "chat": "/chat",
            "chats": "/chats/{user_id}",
            "messages": "/chats/{chat_id}/messages",
            "projects": "/projects/{project_id}/status",
            "research": "/research",
            "generate_image": "/generate/image",
            "generate_video": "/generate/video",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
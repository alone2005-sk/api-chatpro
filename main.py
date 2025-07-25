"""
DAMN BOT - Professional All-in-One AI Chat Agent
FastAPI backend with multi-LLM orchestration, file processing, web search, and voice generation
"""

import os
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
from middleware.rate_limiter import RateLimiter

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global chat_service, llm_orchestrator, file_processor, web_search_service
    global voice_service, code_service, research_service, deep_learning_service, db_manager
    
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
        
        code_service = CodeService(settings)
        await code_service.initialize()
        
        research_service = ResearchService(settings, web_search_service, llm_orchestrator)
        await research_service.initialize()
        
        deep_learning_service = DeepLearningService(settings)
        await deep_learning_service.initialize()
        
        chat_service = ChatService(
            db_manager=db_manager,
            llm_orchestrator=llm_orchestrator,
            file_processor=file_processor,
            web_search_service=web_search_service,
            voice_service=voice_service,
            code_service=code_service,
            research_service=research_service,
            deep_learning_service=deep_learning_service,
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
app.add_middleware(RateLimiter, calls=settings.RATE_LIMIT_CALLS, period=settings.RATE_LIMIT_PERIOD)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    prompt: Optional[str] = Form(None),
    files: List[UploadFile] = File(None),
    web_search: bool = Form(False),
    voice: bool = Form(False),
    project_id: Optional[str] = Form(None),
    stream: bool = Form(False),
    research_mode: bool = Form(False),
    deep_learning: bool = Form(False),
    code_execution: bool = Form(True),
    auto_fix: bool = Form(True),
    language: Optional[str] = Form("auto"),
    max_iterations: int = Form(3)
):
    """
    Unified AI Chat endpoint - handles all AI tasks through a single interface
    
    Features:
    - Multi-LLM orchestration with intelligent response merging
    - File processing (PDF, DOCX, TXT, audio, video, images)
    - Web search integration with live results
    - Voice generation with multiple TTS engines
    - Code generation with auto-testing and fixing
    - Research mode with deep analysis
    - Deep learning integration for specialized tasks
    - Session tracking and history management
    - Streaming responses support
    """
    try:
        # Validate input
        if not prompt and not files:
            raise HTTPException(
                status_code=400,
                detail="Either prompt or files must be provided"
            )
        
        # Create chat request
        chat_request = ChatRequest(
            prompt=prompt or "",
            files=files or [],
            web_search=web_search,
            voice=voice,
            project_id=project_id,
            stream=stream,
            research_mode=research_mode,
            deep_learning=deep_learning,
            code_execution=code_execution,
            auto_fix=auto_fix,
            language=language,
            max_iterations=max_iterations
        )
        
        # Process chat request
        if stream:
            return StreamingResponse(
                chat_service.process_chat_stream(chat_request),
                media_type="text/event-stream"
            )
        else:
            response = await chat_service.process_chat(chat_request)
            
            # Add background tasks for cleanup and learning
            background_tasks.add_task(
                chat_service.post_process_cleanup,
                chat_request.project_id or response.project_id
            )
            
            return response
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def health_check():
    """System health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {}
        }
        
        # Check service health
        if llm_orchestrator:
            status["services"]["llm_orchestrator"] = await llm_orchestrator.health_check()
        if file_processor:
            status["services"]["file_processor"] = await file_processor.health_check()
        if web_search_service:
            status["services"]["web_search"] = await web_search_service.health_check()
        if voice_service:
            status["services"]["voice"] = await voice_service.health_check()
        if code_service:
            status["services"]["code"] = await code_service.health_check()
        if research_service:
            status["services"]["research"] = await research_service.health_check()
        if deep_learning_service:
            status["services"]["deep_learning"] = await deep_learning_service.health_check()
        
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=500
        )

@app.get("/projects/{project_id}/history")
async def get_project_history(project_id: str):
    """Get complete project history and artifacts"""
    try:
        history = await chat_service.get_project_history(project_id)
        return JSONResponse(history)
    except Exception as e:
        logger.error(f"Project history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/download")
async def download_project_artifacts(project_id: str):
    """Download project artifacts as ZIP file"""
    try:
        zip_path = await chat_service.create_project_zip(project_id)
        return FileResponse(
            zip_path,
            filename=f"project_{project_id}.zip",
            media_type="application/zip"
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and statistics"""
    try:
        metrics = await chat_service.get_system_metrics()
        return JSONResponse(metrics)
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get available LLM models and their status"""
    try:
        models = await llm_orchestrator.get_available_models()
        return JSONResponse(models)
    except Exception as e:
        logger.error(f"Models error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

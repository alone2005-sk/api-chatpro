"""
Advanced AI Agent Backend - Main FastAPI Application
Comprehensive AI system with multi-format analysis, project generation, and learning capabilities
"""

import os
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db, init_database
from core.file_processor import FileProcessor
from core.llm_orchestrator import LLMOrchestrator
from core.project_generator import ProjectGenerator
from core.code_tester import CodeTester
from core.learning_engine import LearningEngine
from core.session_manager import SessionManager
from core.quality_scorer import QualityScorer
from models.requests import *
from models.responses import *
from services.chat_service import ChatService
from services.analysis_service import AnalysisService
from services.history_service import HistoryService
from services.fix_service import FixService
from utils.auth import verify_token
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global components
file_processor = FileProcessor()
llm_orchestrator = LLMOrchestrator()
project_generator = ProjectGenerator()
code_tester = CodeTester()
learning_engine = LearningEngine()
session_manager = SessionManager()
quality_scorer = QualityScorer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🚀 Starting Advanced AI Agent Backend...")
    
    # Initialize database
    await init_database()
    
    # Initialize components
    await file_processor.initialize()
    await llm_orchestrator.initialize()
    await project_generator.initialize()
    await code_tester.initialize()
    await learning_engine.initialize()
    
    logger.info("✅ All systems ready!")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down...")
    await llm_orchestrator.cleanup()
    await code_tester.cleanup()
    logger.info("✅ Shutdown complete!")

app = FastAPI(
    title="Advanced AI Agent Backend",
    description="Comprehensive AI system with multi-format analysis and project generation",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
chat_service = ChatService()
analysis_service = AnalysisService()
history_service = HistoryService()
fix_service = FixService()

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Main chat endpoint - handles text prompts and conversations"""
    try:
        # Verify authentication
        user_id = await verify_token(credentials.credentials)
        
        # Process chat request
        response = await chat_service.process_chat(
            request=request,
            user_id=user_id,
            db=db,
            background_tasks=background_tasks
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    project_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """File analysis endpoint - handles multi-format file uploads"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        request = AnalysisRequest(
            files=files,
            prompt=prompt,
            project_id=project_id
        )
        
        response = await analysis_service.analyze_files(
            request=request,
            user_id=user_id,
            db=db,
            background_tasks=background_tasks
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history/{project_id}", response_model=HistoryResponse)
async def get_history(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get complete project/chat history"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        history = await history_service.get_project_history(
            project_id=project_id,
            user_id=user_id,
            db=db
        )
        
        return history
        
    except Exception as e:
        logger.error(f"History endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/fix", response_model=FixResponse)
async def fix_endpoint(
    request: FixRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Automatic error analysis and code fixing"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        response = await fix_service.fix_code_errors(
            request=request,
            user_id=user_id,
            db=db,
            background_tasks=background_tasks
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Fix endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/snapshot", response_model=SnapshotResponse)
async def create_snapshot(
    request: SnapshotRequest,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Save progress snapshots"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        snapshot = await session_manager.create_snapshot(
            request=request,
            user_id=user_id,
            db=db
        )
        
        return snapshot
        
    except Exception as e:
        logger.error(f"Snapshot endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/projects/{project_id}/stream")
async def stream_project_progress(
    project_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Stream real-time project generation progress"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        async def event_stream():
            async for event in session_manager.stream_progress(project_id, user_id):
                yield f"data: {event}\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Stream endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate-project", response_model=ProjectResponse)
async def generate_project(
    request: ProjectRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate complete professional projects"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        # Start project generation in background
        project_id = await project_generator.start_generation(
            request=request,
            user_id=user_id,
            db=db
        )
        
        background_tasks.add_task(
            project_generator.generate_complete_project,
            project_id=project_id,
            request=request,
            user_id=user_id,
            db=db
        )
        
        return ProjectResponse(
            project_id=project_id,
            status="started",
            message="Project generation started"
        )
        
    except Exception as e:
        logger.error(f"Project generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/llm-performance")
async def get_llm_performance(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get LLM performance statistics and rankings"""
    try:
        await verify_token(credentials.credentials)
        
        performance = await quality_scorer.get_performance_stats()
        return JSONResponse(performance)
        
    except Exception as e:
        logger.error(f"Performance endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/learning-insights")
async def get_learning_insights(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get learning engine insights and improvements"""
    try:
        await verify_token(credentials.credentials)
        
        insights = await learning_engine.get_insights()
        return JSONResponse(insights)
        
    except Exception as e:
        logger.error(f"Learning insights error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/system/status")
async def system_status():
    """Get comprehensive system status"""
    try:
        status = {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "file_processor": await file_processor.get_status(),
                "llm_orchestrator": await llm_orchestrator.get_status(),
                "project_generator": await project_generator.get_status(),
                "code_tester": await code_tester.get_status(),
                "learning_engine": await learning_engine.get_status()
            },
            "active_sessions": await session_manager.get_active_count(),
            "total_projects": await history_service.get_total_projects()
        }
        
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"System status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=500, reload=True)

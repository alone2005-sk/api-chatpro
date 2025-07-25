"""
Main chat service orchestrating all AI capabilities
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
import zipfile

from core.database import DatabaseManager, Project, Message, Artifact
from core.logger import get_logger
from models.requests import ChatRequest, ChatResponse
from services.task_detector import TaskDetector
from services.llm_orchestrator import LLMOrchestrator
from services.file_processor import FileProcessor
from services.web_search import WebSearchService
from services.voice_service import VoiceService
from services.code_service import CodeService
from services.research_service import ResearchService
from services.deep_learning_service import DeepLearningService

logger = get_logger(__name__)

class ChatService:
    """Main chat service coordinating all AI capabilities"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        llm_orchestrator: LLMOrchestrator,
        file_processor: FileProcessor,
        web_search_service: WebSearchService,
        voice_service: VoiceService,
        code_service: CodeService,
        research_service: ResearchService,
        deep_learning_service: DeepLearningService,
        settings
    ):
        self.db_manager = db_manager
        self.llm_orchestrator = llm_orchestrator
        self.file_processor = file_processor
        self.web_search_service = web_search_service
        self.voice_service = voice_service
        self.code_service = code_service
        self.research_service = research_service
        self.deep_learning_service = deep_learning_service
        self.settings = settings
        self.task_detector = TaskDetector()
        
        # Ensure project data directory exists
        self.project_data_dir = Path(settings.PROJECT_DATA_DIR)
        self.project_data_dir.mkdir(exist_ok=True)
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request with full AI pipeline"""
        start_time = time.time()
        
        try:
            # Generate or use existing project ID
            project_id = request.project_id or str(uuid.uuid4())
            
            # Create project if needed
            await self._ensure_project_exists(project_id)
            
            # Create project directory
            project_dir = self.project_data_dir / project_id
            project_dir.mkdir(exist_ok=True)
            
            # Log incoming request
            logger.info(f"Processing chat request for project {project_id}")
            
            # Step 1: Process uploaded files
            file_contents = []
            processed_files = []
            
            if request.files:
                logger.info(f"Processing {len(request.files)} uploaded files")
                for file in request.files:
                    try:
                        file_result = await self.file_processor.process_file(
                            file, project_id
                        )
                        file_contents.append(file_result.content)
                        processed_files.append(file_result.dict())
                    except Exception as e:
                        logger.error(f"Error processing file {file.filename}: {str(e)}")
                        continue
            
            # Step 2: Detect task type and intent
            combined_context = request.prompt
            if file_contents:
                combined_context += "\n\nFile contents:\n" + "\n".join(file_contents)
            
            task_info = await self.task_detector.detect_task(
                combined_context, request.dict()
            )
            
            logger.info(f"Detected task: {task_info['task_type']} (confidence: {task_info['confidence']})")
            
            # Step 3: Web search if requested
            search_results = None
            if request.web_search:
                logger.info("Performing web search")
                search_results = await self.web_search_service.search(
                    request.prompt, max_results=10
                )
                if search_results:
                    combined_context += f"\n\nWeb search results:\n{search_results['summary']}"
            
            # Step 4: Research mode if enabled
            research_data = None
            if request.research_mode:
                logger.info("Enabling research mode")
                research_data = await self.research_service.deep_research(
                    request.prompt, combined_context
                )
                if research_data:
                    combined_context += f"\n\nResearch findings:\n{research_data['summary']}"
            
            # Step 5: Deep learning processing if enabled
            if request.deep_learning:
                logger.info("Applying deep learning processing")
                dl_result = await self.deep_learning_service.process(
                    combined_context, task_info
                )
                if dl_result:
                    combined_context += f"\n\nDeep learning insights:\n{dl_result['analysis']}"
            
            # Step 6: Generate response using multi-LLM orchestration
            logger.info("Generating response with multi-LLM orchestration")
            llm_response = await self.llm_orchestrator.generate_response(
                prompt=combined_context,
                task_type=task_info['task_type'],
                context={
                    'files': processed_files,
                    'search_results': search_results,
                    'research_data': research_data,
                    'project_id': project_id
                }
            )
            
            # Step 7: Code handling if detected
            code_result = None
            if task_info['task_type'] in ['code_generation', 'code_fix', 'code_review']:
                logger.info("Processing code-related task")
                code_result = await self.code_service.handle_code_task(
                    llm_response['text'],
                    task_info,
                    project_id,
                    request.code_execution,
                    request.auto_fix,
                    request.max_iterations
                )
            
            # Step 8: Voice generation if requested
            audio_file = None
            if request.voice:
                logger.info("Generating voice output")
                voice_result = await self.voice_service.generate_speech(
                    llm_response['text'], project_id
                )
                audio_file = voice_result.get('audio_file')
            
            # Step 9: Save conversation to database
            message_id = await self._save_conversation(
                project_id,
                request.prompt,
                llm_response['text'],
                {
                    'files': processed_files,
                    'search_results': search_results,
                    'research_data': research_data,
                    'code_result': code_result,
                    'task_info': task_info,
                    'llm_scores': llm_response.get('scores', {})
                }
            )
            
            # Step 10: Prepare response
            processing_time = time.time() - start_time
            
            response = ChatResponse(
                project_id=project_id,
                message_id=message_id,
                text=llm_response['text'],
                code=code_result.get('code') if code_result else None,
                language=code_result.get('language') if code_result else None,
                files=code_result.get('files', []) if code_result else [],
                audio_file=audio_file,
                sources=search_results.get('sources', []) if search_results else [],
                research_data=research_data,
                execution_results=code_result.get('execution_results') if code_result else None,
                llm_scores=llm_response.get('scores', {}),
                processing_time=processing_time,
                metadata={
                    'task_type': task_info['task_type'],
                    'confidence': task_info['confidence'],
                    'processed_files': len(processed_files),
                    'web_search_enabled': request.web_search,
                    'voice_enabled': request.voice,
                    'research_mode': request.research_mode,
                    'deep_learning': request.deep_learning
                }
            )
            
            logger.info(f"Chat processing completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Chat processing error: {str(e)}")
            raise
    
    async def process_chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Process chat request with streaming response"""
        project_id = request.project_id or str(uuid.uuid4())
        
        try:
            yield f"data: {json.dumps({'type': 'start', 'project_id': project_id})}\n\n"
            
            # Process files
            if request.files:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Processing files...'})}\n\n"
                # File processing logic here
            
            # Web search
            if request.web_search:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching web...'})}\n\n"
                # Web search logic here
            
            # Generate response
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            
            # Stream LLM response
            async for chunk in self.llm_orchestrator.generate_response_stream(
                request.prompt, 'chat'
            ):
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'end', 'project_id': project_id})}\n\n"
            
        except Exception as e:
            error_msg = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield error_msg
    
    async def _ensure_project_exists(self, project_id: str):
        """Ensure project exists in database"""
        async with self.db_manager.get_session() as session:
            # Check if project exists
            project = await session.get(Project, project_id)
            if not project:
                # Create new project
                project = Project(
                    id=project_id,
                    name=f"Project {project_id[:8]}",
                    description="AI Chat Session",
                    created_at=datetime.utcnow(),
                    status="active"
                )
                session.add(project)
                await session.commit()
    
    async def _save_conversation(
        self,
        project_id: str,
        user_message: str,
        ai_response: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Save conversation to database"""
        async with self.db_manager.get_session() as session:
            # Save user message
            user_msg = Message(
                project_id=project_id,
                role="user",
                content=user_message,
                timestamp=datetime.utcnow(),
                message_type="text",
                metadata=metadata
            )
            session.add(user_msg)
            
            # Save AI response
            ai_msg = Message(
                project_id=project_id,
                role="assistant",
                content=ai_response,
                timestamp=datetime.utcnow(),
                message_type="text",
                metadata=metadata
            )
            session.add(ai_msg)
            
            await session.commit()
            return ai_msg.id
    
    async def get_project_history(self, project_id: str) -> Dict[str, Any]:
        """Get complete project history"""
        async with self.db_manager.get_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Get all messages
            messages = await session.query(Message).filter(
                Message.project_id == project_id
            ).order_by(Message.timestamp).all()
            
            # Get all artifacts
            artifacts = await session.query(Artifact).filter(
                Artifact.project_id == project_id
            ).all()
            
            return {
                'project': {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'created_at': project.created_at.isoformat(),
                    'status': project.status
                },
                'messages': [
                    {
                        'id': msg.id,
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'type': msg.message_type,
                        'metadata': msg.metadata
                    }
                    for msg in messages
                ],
                'artifacts': [
                    {
                        'id': art.id,
                        'name': art.name,
                        'type': art.type,
                        'file_path': art.file_path,
                        'size': art.size,
                        'created_at': art.created_at.isoformat()
                    }
                    for art in artifacts
                ]
            }
    
    async def create_project_zip(self, project_id: str) -> str:
        """Create ZIP file of project artifacts"""
        project_dir = self.project_data_dir / project_id
        if not project_dir.exists():
            raise ValueError(f"Project {project_id} not found")
        
        zip_path = project_dir / f"project_{project_id}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in project_dir.rglob('*'):
                if file_path.is_file() and file_path != zip_path:
                    arcname = file_path.relative_to(project_dir)
                    zipf.write(file_path, arcname)
        
        return str(zip_path)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics and statistics"""
        async with self.db_manager.get_session() as session:
            # Get project count
            project_count = await session.query(Project).count()
            
            # Get message count
            message_count = await session.query(Message).count()
            
            # Get recent activity
            recent_projects = await session.query(Project).order_by(
                Project.updated_at.desc()
            ).limit(10).all()
            
            return {
                'total_projects': project_count,
                'total_messages': message_count,
                'recent_projects': [
                    {
                        'id': p.id,
                        'name': p.name,
                        'updated_at': p.updated_at.isoformat()
                    }
                    for p in recent_projects
                ],
                'system_status': 'healthy',
                'uptime': time.time(),
                'services': await self._get_service_status()
            }
    
    async def _get_service_status(self) -> Dict[str, str]:
        """Get status of all services"""
        return {
            'llm_orchestrator': 'healthy',
            'file_processor': 'healthy',
            'web_search': 'healthy',
            'voice_service': 'healthy',
            'code_service': 'healthy',
            'research_service': 'healthy',
            'deep_learning': 'healthy'
        }
    
    async def post_process_cleanup(self, project_id: str):
        """Background cleanup tasks"""
        try:
            # Clean up temporary files
            temp_dir = Path(self.settings.TEMP_DIR) / project_id
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            
            logger.info(f"Cleanup completed for project {project_id}")
            
        except Exception as e:
            logger.error(f"Cleanup error for project {project_id}: {str(e)}")

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
from services.task_detector import TaskDetector, TaskType
from services.llm_orchestrator import LLMOrchestrator
from services.file_processor import FileProcessor
from services.web_search import WebSearchService    
import os
import logging
from fastapi import HTTPException, BackgroundTasks, Form
from pathlib import Path
import zipfile
import logging
from services.self_learning import SelfLearningService
from services.project_generator import ProjectGenerationService
from models.requests import UnifiedChatRequest, EnhancedChatResponse
from core.config import settings
from core.database import DatabaseManager, Chat, Message, MediaFile
from services.image_video_generator import ImageVideoGenerator
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
class ChatService:
    """Core chat processing service with AI capabilities"""

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
        project_generation_service: ProjectGenerationService,
        self_learning_service: SelfLearningService,
        image_video_generator: ImageVideoGenerator,
        settings
    ):
        self.db_manager = db_manager
        self.llm_orchestrator = llm_orchestrator
        self.file_processor = file_processor
        self.web_search = web_search_service
        self.voice_service = voice_service
        self.code_service = code_service
        self.research_service = research_service
        self.deep_learning = deep_learning_service
        self.project_service = project_generation_service
        self.learning_service = self_learning_service
        self.media_generator = image_video_generator
        self.task_detector = TaskDetector()

        # Ensure project data directory exists
        self.project_data_dir = Path(settings.PROJECT_DATA_DIR)
        self.project_data_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
    
    async def process_chat(self, request: UnifiedChatRequest, background_tasks) -> EnhancedChatResponse:
        """Process chat request with full AI pipeline"""
        start_time = time.time()
        
        # Create or get chat session
        if request.chat_id:
            chat = await self.db_manager.get_chat(request.chat_id)
            if not chat:
                raise HTTPException(status_code=404, detail="Chat session not found")
        else:
            title = request.prompt[:50] + "..." if len(request.prompt) > 50 else request.prompt
            chat = await self.db_manager.create_chat(
                user_id=request.user_id,
                title=title,
                description="AI Agent Session"
            )
        
        # Process files
        file_contents = []
        processed_files = []
        if request.files:
            for file in request.files:
                try:
                    file_result = await self.file_processor.process_file(file, chat.id)
                    file_contents.append(file_result.content)
                    processed_files.append(file_result.dict())
                except Exception as e:
                    self.logger.error(f"Error processing file {file.filename}: {str(e)}")
        
        # Build combined context
        combined_context = request.prompt
        if file_contents:
            combined_context += "\n\nFile contents:\n" + "\n".join(file_contents)
        
        # Advanced task detection with new TaskDetector
        task_info = await self.task_detector.analyze_intent(combined_context, request.dict())
        self.logger.info(
            f"Detected task: {task_info.task_type} "
            f"(confidence: {task_info.confidence:.2f}, "
            f"complexity: {task_info.complexity_level}, "
            f"language: {task_info.language})"
        )
        
        # Intent detection pipeline
        # 1. Media generation (prioritize media tasks)
        media_enabled = (
            request.auto_detect_media and 
            settings.ENABLE_MEDIA_GENERATION and
            task_info.task_type in (TaskType.IMAGE_CREATION, TaskType.VIDEO_CREATION)
        )
        if media_enabled:
            media_result = await self.media_generator.detect_and_generate_media(
                combined_context, 
                request.user_id,
                task_info=task_info  # Pass task metadata
            )
            if media_result.get("is_media_request") and media_result.get("success"):
                return await self.handle_media_generation(
                    media_result, 
                    chat.id, 
                    request, 
                    start_time, 
                    background_tasks
                )
        
        # 2. Project generation
        project_enabled = (
            request.auto_detect_project and 
            settings.ENABLE_PROJECT_GENERATION and
            task_info.task_type == TaskType.PROJECT_CREATION
        )
        if project_enabled:
            project_result = await self.project_service.detect_and_create_project(
                combined_context, 
                chat.id, 
                request.user_id,
                task_info=task_info  # Pass task metadata
            )
            if project_result.get("is_project"):
                return await self.handle_project_generation(
                    project_result, 
                    chat.id, 
                    request, 
                    start_time, 
                    background_tasks
                )
        
        # 3. Self-learning
        learning_result = None
        if request.enable_learning and settings.ENABLE_SELF_LEARNING:
            learning_result = await self.learning_service.generate_improved_response(
                combined_context, 
                {"chat_id": chat.id, **request.context},
                request.user_id
            )
            if learning_result.get("success") and learning_result.get("confidence", 0) > 0.7:
                return await self.handle_learning_response(
                    learning_result, 
                    chat.id, 
                    request, 
                    start_time
                )
        
        # AI processing pipeline
        # 4. Web search
        search_results = None
        if request.web_search:
            search_results = await self.web_search.search(request.prompt, max_results=10)
            if search_results:
                combined_context += f"\n\nSearch Results:\n{search_results['summary']}"
        
        # 5. Deep research (enhanced for media/research tasks)
        research_data = None
        if request.enable_research and settings.ENABLE_DEEP_RESEARCH:
            research_depth = "deep" if task_info.task_type == TaskType.RESEARCH else request.research_depth
            research_result = await self.research_service.conduct_research(
                combined_context, 
                chat.id, 
                research_depth, 
                {**request.context, "task_info": task_info.dict()}  # Include task metadata
            )
            if not research_result.get("error"):
                research_data = research_result
                if research_result.get("report", {}).get("report_text", ""):
                    combined_context += f"\n\nResearch Context:\n{research_result['report']['report_text'][:500]}..."
        
        # 6. Deep learning processing
        if request.deep_learning:
            dl_result = await self.deep_learning.process(combined_context, task_info)
            if dl_result:
                combined_context += f"\n\nDeep Insights:\n{dl_result['analysis']}"
        
        # 7. Generate LLM response with task context
        llm_response = await self.llm_orchestrator.generate_response(
            prompt=combined_context,
            #task_info=task_info,  # Full task metadata
            context={
                "chat_id": chat.id,
                "user_id": request.user_id,
                "research_available": bool(research_data),
                "files": processed_files,
                "search_results": search_results,
                "research_data": research_data,
                **request.context
            },
            user_id=request.user_id,
            prefer_local=request.prefer_local
        )
        
        # 8. Handle code tasks with enhanced parameters
        code_result = None
        code_task_types = [
            TaskType.CODE_GENERATION, 
            TaskType.BUG_FIXING, 
            TaskType.CODE_REVIEW,
            TaskType.AUTOMATION
        ]
        if task_info.task_type in code_task_types:
            code_result = await self.code_service.handle_code_task(
                llm_response['text'],
                task_info,  # Full task metadata
                chat.id,
                request.code_execution,
                request.auto_fix,
                request.max_iterations
            )
        
        # 9. Generate voice
        audio_file = None
        if request.voice:
            voice_result = await self.voice_service.generate_speech(llm_response['text'], chat.id)
            audio_file = voice_result.get('audio_file')
        
        # 10. Save and return response
        return await self.save_and_return_response(
            llm_response,
            chat.id,
            request,
            start_time,
            background_tasks,
            code_result=code_result,
            audio_file=audio_file,
            research_data=research_data,
            search_results=search_results,
            learning_result=learning_result,
            # task_info=task_info  # Include task metadata in response
        )
    
    async def process_chat_stream(self, request: UnifiedChatRequest) -> AsyncGenerator[str, None]:
        """Process chat request with streaming response"""
        try:
            yield json.dumps({"type": "status", "message": "Starting processing..."})
            
            # Simplified streaming logic
            async for chunk in self.llm_orchestrator.generate_response_stream(
                request.prompt, 
                "conversational",
                context={"user_id": request.user_id}
            ):
                yield json.dumps({"type": "text", "content": chunk})
            
            yield json.dumps({"type": "complete"})
        
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)})
    
    async def handle_project_generation(self, project_result, chat_id, request, start_time, bg_tasks):
        """Handle project generation response"""
        ai_message = await self.db_manager.save_message(
            chat_id=chat_id,
            role="assistant",
            content=f"🚀 Generating {project_result['project_type']} project: {project_result['project_name']}",
            message_type="project",
            meta=project_result
        )
        
        if request.enable_learning:
            bg_tasks.add_task(
                self.learning_service.learn_from_interaction,
                request.user_id,
                chat_id,
                request.prompt,
                ai_message.content,
                None,
                {"task_type": "project_generation"}
            )
        
        return EnhancedChatResponse(
            success=True,
            chat_id=chat_id,
            message_id=ai_message.id,
            response=ai_message.content,
            response_type="project",
            project_id=project_result.get("project_id"),
            processing_time=time.time() - start_time,
            confidence_score=project_result.get("confidence", 0.0)
        )
    
    async def handle_media_generation(self, media_result, chat_id, request, start_time, bg_tasks):
        """Handle media generation response"""
        media_files = []
        if media_result.get("image_path"):
            media_file = await self.db_manager.save_media_file(
                chat_id=chat_id,
                filename=os.path.basename(media_result["image_path"]),
                file_path=media_result["image_path"],
                file_type="image",
                prompt_used=request.prompt
            )
            media_files.append({
                "id": media_file.id,
                "type": "image",
                "url": f"/media/{media_file.filename}"
            })
        
        ai_message = await self.db_manager.save_message(
            chat_id=chat_id,
            role="assistant",
            content=f"🎨 Generated {media_result.get('media_type', 'media')}",
            message_type="media",
            meta=media_result
        )
        
        if request.enable_learning:
            bg_tasks.add_task(
                self.learning_service.learn_from_interaction,
                request.user_id,
                chat_id,
                request.prompt,
                ai_message.content,
                None,
                {"task_type": "media_generation"}
            )
        
        return EnhancedChatResponse(
            success=True,
            chat_id=chat_id,
            message_id=ai_message.id,
            response=ai_message.content,
            response_type="media",
            media_files=media_files,
            processing_time=time.time() - start_time
        )
    
    async def save_and_return_response(self, llm_response, chat_id, request, start_time, bg_tasks, 
                                      code_result=None, audio_file=None, research_data=None,
                                      search_results=None, learning_result=None):
        content = llm_response.get("text")
        if content is None:
               self.logger.error("LLM response content is None")
               raise HTTPException(status_code=500, detail="LLM response missing 'text' field.")
        """Save final response and return"""
        ai_message = await self.db_manager.save_message(
            chat_id=chat_id,
            role="assistant",
            content=content,
            message_type="text",
            meta={
                "model_used": llm_response.get("model"),
                "tokens_used": llm_response.get("tokens_used", 0)
            }
        )
        
        if request.enable_learning:
            bg_tasks.add_task(
                self.learning_service.learn_from_interaction,
                request.user_id,
                chat_id,
                request.prompt,
                ai_message.content,
                None,
                {"task_type": "chat", "model": llm_response.get("model")}
            )
        
        return EnhancedChatResponse(
            success=True,
            chat_id=chat_id,
            message_id=ai_message.id,
            response=ai_message.content,
            response_type="text",
            model_used=llm_response.get("model"),
            processing_time=time.time() - start_time,
            tokens_used=llm_response.get("tokens_used", 0),
            confidence_score=llm_response.get("confidence", 0.0),
            code=code_result.get("code") if code_result else None,
            language=code_result.get("language") if code_result else None,
            audio_file=audio_file,
            research_data=research_data,
            sources=search_results.get("results", [])[:5] if search_results else [],
            learning_applied=bool(learning_result and learning_result.get("success")),
            llm_scores=llm_response.get("scores", {}),
            execution_results=code_result.get("execution_results") if code_result else None,
            metadata={
                "processed_files": len(request.files),
                "research_performed": bool(research_data)
            }
        )
    
    async def get_user_chats(self, user_id: str, limit: int = 50):
        """Get user's chat sessions"""
        chats = await self.db_manager.get_user_chats(user_id, limit)
        return {
            "success": True,
            "chats": [chat.to_dict() for chat in chats]
        }
    
    async def get_chat_messages(self, chat_id: str, limit: int = 50, offset: int = 0):
        """Get chat messages with pagination"""
        messages = await self.db_manager.get_chat_messages(chat_id, limit, offset)
        return {
            "success": True,
            "chat_id": chat_id,
            "messages": [msg.to_dict() for msg in messages]
        }
    
    async def delete_chat(self, chat_id: str):
        """Delete a chat session"""
        await self.db_manager.update_chat(chat_id, is_active=False)
        return {
            "success": True,
            "message": "Chat session deleted successfully"
        }
    
    async def get_project_status(self, project_id: str):
        """Get project generation status"""
        status = await self.project_service.get_project_status(project_id)
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        return status
    
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
            
            self.logger.info(f"Cleanup completed for project {project_id}")
            
        except Exception as e:
            self.logger.error(f"Cleanup error for project {project_id}: {str(e)}")

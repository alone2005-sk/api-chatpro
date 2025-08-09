import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, String, DateTime, Integer, Text, Boolean, JSON, ForeignKey, Float
)
from sqlalchemy.orm import relationship, declarative_base, selectinload
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession
)
from sqlalchemy import select, desc, and_
from contextlib import asynccontextmanager

Base = declarative_base()

class Chat(Base):
    """Chat session model"""
    __tablename__ = "chats"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    meta = Column(JSON, default=dict)
    
    # Relationships
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    """Message model for chat history"""
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.id"), nullable=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(String, default="text")
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    model_used = Column(String)
    tokens_used = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    
    # Relationships
    chat = relationship("Chat", back_populates="messages")
    project = relationship("Project", back_populates="messages")
    media_files = relationship("MediaFile", back_populates="message", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="message", cascade="all, delete-orphan")

class Project(Base):
    """Project model for generated projects"""
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.id"), nullable=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    project_type = Column(String, nullable=False)
    status = Column(String, default="generating")
    progress = Column(Integer, default=0)
    file_path = Column(String)
    zip_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    meta = Column(JSON, default=dict)
    
    # Relationships
    chat = relationship("Chat", back_populates="projects")
    messages = relationship("Message", back_populates="project", cascade="all, delete-orphan")
    files = relationship("ProjectFile", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")
    llm_scores = relationship("LLMScore", back_populates="project", cascade="all, delete-orphan")
    research_results = relationship("ResearchResult", back_populates="project", cascade="all, delete-orphan")
    code_executions = relationship("CodeExecution", back_populates="project", cascade="all, delete-orphan")

class ProjectFile(Base):
    """Project file model"""
    __tablename__ = "project_files"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    size = Column(Integer, default=0)
    content_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="files")

class MediaFile(Base):
    """Media file model"""
    __tablename__ = "media_files"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String, ForeignKey("messages.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    mime_type = Column(String)
    size = Column(Integer, default=0)
    width = Column(Integer)
    height = Column(Integer)
    duration = Column(Float)
    generated_by = Column(String)
    prompt_used = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    
    # Relationships
    message = relationship("Message", back_populates="media_files")

class ResearchSession(Base):
    """Research session model for deep research"""
    __tablename__ = "research_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.id"), nullable=False)
    query = Column(Text, nullable=False)
    research_type = Column(String, default="comprehensive")
    status = Column(String, default="processing")
    sources_found = Column(Integer, default=0)
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    results = Column(JSON, default=dict)
    
    # Relationships
    sources = relationship("ResearchSource", back_populates="session", cascade="all, delete-orphan")

class ResearchSource(Base):
    """Research source model"""
    __tablename__ = "research_sources"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=False)
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    content = Column(Text)
    source_type = Column(String, default="web")
    relevance_score = Column(Float, default=0.0)
    credibility_score = Column(Float, default=0.0)
    extracted_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    
    # Relationships
    session = relationship("ResearchSession", back_populates="sources")

class ResearchResult(Base):
    """Research result model"""
    __tablename__ = "research_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    query = Column(String, nullable=False)
    source = Column(String, nullable=False)
    results = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="research_results")


class CodeExecution(Base):
    """Code execution model"""
    __tablename__ = "code_executions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    code = Column(Text, nullable=False)
    language = Column(String, nullable=False)
    output = Column(Text)
    error = Column(Text)
    exit_code = Column(Integer)
    execution_time = Column(Float)
    success = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="code_executions")

class LearningInteraction(Base):
    """Learning interaction model for self-learning"""
    __tablename__ = "learning_interactions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    chat_id = Column(String, ForeignKey("chats.id"))
    user_input = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    user_feedback = Column(JSON)
    context = Column(JSON, default=dict)
    patterns = Column(JSON, default=dict)
    quality_score = Column(Float, default=0.5)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class LLMScore(Base):
    """LLM score model"""
    __tablename__ = "llm_scores"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    model_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    response_time = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=False)
    success = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="llm_scores")

class Artifact(Base):
    """Artifact model"""
    __tablename__ = "artifacts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    message_id = Column(String, ForeignKey("messages.id"), nullable=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    file_path = Column(String)
    content = Column(Text)
    size = Column(Integer)
    mime_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="artifacts")
    message = relationship("Message", back_populates="artifacts")

class APIUsage(Base):
    """API usage tracking"""
    __tablename__ = "api_usage"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    tokens_used = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    response_time = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database manager for AI Agent Backend"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.async_session = None
    
    async def initialize(self):
        """Initialize database tables"""
        if self.database_url.startswith("sqlite:///"):
            async_url = self.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        else:
            async_url = self.database_url
        
        self.engine = create_async_engine(async_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_chat(self, user_id: str, title: str, description: str = "") -> Chat:
        """Create new chat session"""
        async with self.get_session() as session:
            chat = Chat(
                user_id=user_id,
                title=title,
                description=description
            )
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
            return chat
    
    async def get_chat(self, chat_id: str) -> Optional[Chat]:
        """Get chat by ID with messages"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Chat)
                .options(selectinload(Chat.messages), selectinload(Chat.projects))
                .where(Chat.id == chat_id)
            )
            return result.scalar_one_or_none()
    
    async def get_user_chats(self, user_id: str, limit: int = 50) -> List[Chat]:
        """Get user's chat sessions"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Chat)
                .where(and_(Chat.user_id == user_id, Chat.is_active == True))
                .order_by(desc(Chat.updated_at))
                .limit(limit)
            )
            return result.scalars().all()
    
    async def update_chat(self, chat_id: str, **kwargs) -> Optional[Chat]:
        """Update chat session"""
        async with self.get_session() as session:
            chat = await session.get(Chat, chat_id)
            if chat:
                for key, value in kwargs.items():
                    setattr(chat, key, value)
                chat.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(chat)
            return chat
    
    async def save_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        project_id: str = None,
        meta: Dict[str, Any] = None,
        model_used: str = None,
        tokens_used: int = 0,
        processing_time: float = 0.0,
        confidence_score: float = 0.0
    ) -> Message:
        """Save message to chat"""
        async with self.get_session() as session:
            message = Message(
                chat_id=chat_id,
                project_id=project_id,
                role=role,
                content=content,
                message_type=message_type,
                meta=meta or {},
                model_used=model_used,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            
            # Update chat timestamp
            if chat_id:
                await self.update_chat(chat_id, updated_at=datetime.utcnow())
            
            return message
    
    async def get_chat_messages(
        self,
        chat_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Message]:
        """Get chat messages with pagination"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Message)
                .options(selectinload(Message.media_files), selectinload(Message.artifacts))
                .where(Message.chat_id == chat_id)
                .order_by(desc(Message.timestamp))
                .offset(offset)
                .limit(limit)
            )
            return result.scalars().all()
    
    async def create_project(
        self,
        chat_id: str,
        name: str,
        description: str,
        project_type: str,
        meta: Dict[str, Any] = None
    ) -> Project:
        """Create new project"""
        async with self.get_session() as session:
            project = Project(
                chat_id=chat_id,
                name=name,
                description=description,
                project_type=project_type,
                meta=meta or {}
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)
            return project
    
    async def update_project(self, project_id: str, **kwargs) -> Optional[Project]:
        """Update project"""
        async with self.get_session() as session:
            project = await session.get(Project, project_id)
            if project:
                for key, value in kwargs.items():
                    setattr(project, key, value)
                project.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(project)
            return project
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Project)
                .options(
                    selectinload(Project.files),
                    selectinload(Project.artifacts),
                    selectinload(Project.llm_scores),
                    selectinload(Project.research_results),
                    selectinload(Project.code_executions)
                )
                .where(Project.id == project_id)
            )
            return result.scalar_one_or_none()
    
    async def save_media_file(
        self,
        message_id: str,
        filename: str,
        file_path: str,
        file_type: str,
        **kwargs
    ) -> MediaFile:
        """Save media file"""
        async with self.get_session() as session:
            media_file = MediaFile(
                message_id=message_id,
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                **kwargs
            )
            session.add(media_file)
            await session.commit()
            await session.refresh(media_file)
            return media_file
    
    async def create_research_session(
        self,
        chat_id: str,
        query: str,
        research_type: str = "comprehensive"
    ) -> ResearchSession:
        """Create research session"""
        async with self.get_session() as session:
            research_session = ResearchSession(
                chat_id=chat_id,
                query=query,
                research_type=research_type
            )
            session.add(research_session)
            await session.commit()
            await session.refresh(research_session)
            return research_session
    
    async def save_research_source(
        self,
        session_id: str,
        title: str,
        url: str,
        content: str,
        **kwargs
    ) -> ResearchSource:
        """Save research source"""
        async with self.get_session() as session:
            source = ResearchSource(
                session_id=session_id,
                title=title,
                url=url,
                content=content,
                **kwargs
            )
            session.add(source)
            await session.commit()
            await session.refresh(source)
            return source
    
    async def save_learning_interaction(
        self,
        user_id: str,
        chat_id: str,
        user_input: str,
        ai_response: str,
        **kwargs
    ) -> LearningInteraction:
        """Save learning interaction"""
        async with self.get_session() as session:
            interaction = LearningInteraction(
                user_id=user_id,
                chat_id=chat_id,
                user_input=user_input,
                ai_response=ai_response,
                **kwargs
            )
            session.add(interaction)
            await session.commit()
            await session.refresh(interaction)
            return interaction
    
    async def get_unprocessed_learning_interactions(self, limit: int = 100) -> List[LearningInteraction]:
        """Get unprocessed learning interactions"""
        async with self.get_session() as session:
            result = await session.execute(
                select(LearningInteraction)
                .where(LearningInteraction.processed == False)
                .order_by(LearningInteraction.created_at)
                .limit(limit)
            )
            return result.scalars().all()
    
    async def mark_learning_interactions_processed(self, interaction_ids: List[str]):
        """Mark learning interactions as processed"""
        async with self.get_session() as session:
            await session.execute(
                LearningInteraction.__table__.update()
                .where(LearningInteraction.id.in_(interaction_ids))
                .values(processed=True)
            )
            await session.commit()
    
    async def track_api_usage(
        self,
        user_id: str,
        provider: str,
        model: str,
        endpoint: str,
        **kwargs
    ):
        """Track API usage"""
        async with self.get_session() as session:
            usage = APIUsage(
                user_id=user_id,
                provider=provider,
                model=model,
                endpoint=endpoint,
                **kwargs
            )
            session.add(usage)
            await session.commit()
    
    async def save_artifact(
        self,
        project_id: str,
        message_id: str,
        name: str,
        type: str,
        **kwargs
    ) -> Artifact:
        """Save artifact"""
        async with self.get_session() as session:
            artifact = Artifact(
                project_id=project_id,
                message_id=message_id,
                name=name,
                type=type,
                **kwargs
            )
            session.add(artifact)
            await session.commit()
            await session.refresh(artifact)
            return artifact
    
    async def save_research_result(
        self,
        project_id: str,
        query: str,
        source: str,
        results: Dict,
        **kwargs
    ) -> ResearchResult:
        """Save research result"""
        async with self.get_session() as session:
            research_result = ResearchResult(
                project_id=project_id,
                query=query,
                source=source,
                results=results,
                **kwargs
            )
            session.add(research_result)
            await session.commit()
            await session.refresh(research_result)
            return research_result
    
    async def save_code_execution(
        self,
        project_id: str,
        code: str,
        language: str,
        **kwargs
    ) -> CodeExecution:
        """Save code execution"""
        async with self.get_session() as session:
            code_execution = CodeExecution(
                project_id=project_id,
                code=code,
                language=language,
                **kwargs
            )
            session.add(code_execution)
            await session.commit()
            await session.refresh(code_execution)
            return code_execution
    
    async def save_llm_score(
        self,
        project_id: str,
        model_name: str,
        task_type: str,
        response_time: float,
        quality_score: float,
        **kwargs
    ) -> LLMScore:
        """Save LLM score"""
        async with self.get_session() as session:
            llm_score = LLMScore(
                project_id=project_id,
                model_name=model_name,
                task_type=task_type,
                response_time=response_time,
                quality_score=quality_score,
                **kwargs
            )
            session.add(llm_score)
            await session.commit()
            await session.refresh(llm_score)
            return llm_score
    
    async def close(self):
        """Cleanup database connections"""
        if self.engine:
            await self.engine.dispose()

async def init_database(db_manager: DatabaseManager):
    """Initialize database"""
    await db_manager.initialize()
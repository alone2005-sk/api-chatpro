"""
Database models and management for DAMN BOT
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship, sessionmaker
import uuid

Base = declarative_base()

class Project(Base):
    """Project/Session model"""
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="active")
    metadata = Column(JSON, default=dict)
    
    # Relationships
    messages = relationship("Message", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")
    llm_scores = relationship("LLMScore", back_populates="project", cascade="all, delete-orphan")

class Message(Base):
    """Chat message model"""
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String, default="text")  # text, code, image, audio, file
    metadata = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="messages")
    artifacts = relationship("Artifact", back_populates="message")

class Artifact(Base):
    """File and generated content artifacts"""
    __tablename__ = "artifacts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    message_id = Column(String, ForeignKey("messages.id"), nullable=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # file, code, audio, image, video
    file_path = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    size = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="artifacts")
    message = relationship("Message", back_populates="artifacts")

class LLMScore(Base):
    """LLM performance tracking"""
    __tablename__ = "llm_scores"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    model_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    response_time = Column(Integer, nullable=False)  # milliseconds
    quality_score = Column(Integer, nullable=False)  # 1-10
    success = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("Project", back_populates="llm_scores")

class ResearchResult(Base):
    """Research and web search results"""
    __tablename__ = "research_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    query = Column(String, nullable=False)
    source = Column(String, nullable=False)  # serpapi, duckduckgo, academic
    results = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)

class CodeExecution(Base):
    """Code execution history"""
    __tablename__ = "code_executions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    code = Column(Text, nullable=False)
    language = Column(String, nullable=False)
    output = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    exit_code = Column(Integer, nullable=True)
    execution_time = Column(Integer, nullable=True)  # milliseconds
    success = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.async_session = None
        
    async def initialize(self):
        """Initialize database connection"""
        # Convert SQLite URL for async
        if self.database_url.startswith("sqlite:///"):
            async_url = self.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        else:
            async_url = self.database_url
            
        self.engine = create_async_engine(async_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession)
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        return self.async_session()
    
    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()

async def init_database(db_manager: DatabaseManager):
    """Initialize database"""
    await db_manager.initialize()

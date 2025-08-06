"""
Database models and management for DAMN BOT
"""

import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy import (
    Column, String, DateTime, Integer, Text, Boolean, JSON, ForeignKey
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession
)

Base = declarative_base()


# === MODELS ===

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="active")
    meta = Column(JSON, default=dict)

    messages = relationship("Message", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")
    llm_scores = relationship("LLMScore", back_populates="project", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String, default="text")
    meta = Column(JSON, default=dict)

    project = relationship("Project", back_populates="messages")
    artifacts = relationship("Artifact", back_populates="message")


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    message_id = Column(String, ForeignKey("messages.id"), nullable=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    size = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)

    project = relationship("Project", back_populates="artifacts")
    message = relationship("Message", back_populates="artifacts")


class LLMScore(Base):
    __tablename__ = "llm_scores"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    model_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    response_time = Column(Integer, nullable=False)
    quality_score = Column(Integer, nullable=False)
    success = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)

    project = relationship("Project", back_populates="llm_scores")


class ResearchResult(Base):
    __tablename__ = "research_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    query = Column(String, nullable=False)
    source = Column(String, nullable=False)
    results = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)


class CodeExecution(Base):
    __tablename__ = "code_executions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    code = Column(Text, nullable=False)
    language = Column(String, nullable=False)
    output = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    exit_code = Column(Integer, nullable=True)
    execution_time = Column(Integer, nullable=True)
    success = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)


# === DB Manager ===

class DatabaseManager:
    """Async DB connection and session manager"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.async_session = None

    async def initialize(self):
        """Setup engine, session, and create tables"""
        if self.database_url.startswith("sqlite:///"):
            async_url = self.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        else:
            async_url = self.database_url

        self.engine = create_async_engine(async_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Provide a transactional scope for DB operations"""
        async with self.async_session() as session:
            yield session

    async def close(self):
        """Dispose engine"""
        if self.engine:
            await self.engine.dispose()


# Optional: simple helper to init at startup
async def init_database(db_manager: DatabaseManager):
    await db_manager.initialize()

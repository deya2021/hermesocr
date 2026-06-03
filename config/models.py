from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import enum
from config.database import Base

class SourceType(str, enum.Enum):
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    GEMINI = "gemini"
    NOTEBOOKLM = "notebooklm"
    OTHER = "other"

class Conversation(Base):
    """المحادثة الكاملة"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    source = Column(Enum(SourceType), nullable=False)
    title = Column(String(500))
    original_id = Column(String(200))          # ID من المصدر الأصلي
    created_at = Column(DateTime, default=datetime.utcnow)
    imported_at = Column(DateTime, default=datetime.utcnow)
    raw_file = Column(String(500))             # مسار الملف الأصلي
    metadata_ = Column("metadata", JSON, default={})
    is_digested = Column(Boolean, default=False, nullable=False)  # تم معالجته في Dream أم لا
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete")
    chunks = relationship("Chunk", back_populates="conversation", cascade="all, delete")

class Message(Base):
    """رسالة واحدة داخل المحادثة"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(20))                  # user / assistant
    content = Column(Text)
    created_at = Column(DateTime)
    order_index = Column(Integer)
    
    conversation = relationship("Conversation", back_populates="messages")

class Chunk(Base):
    """قطعة نص مع embedding للبحث الدلالي"""
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    content = Column(Text)
    embedding = Column(Vector(768))            # nomic-embed-text dimension
    chunk_type = Column(String(50))            # question / answer / insight
    topics = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="chunks")

class WikiPage(Base):
    """صفحة wiki ناتجة"""
    __tablename__ = "wiki_pages"
    
    id = Column(Integer, primary_key=True)
    slug = Column(String(300), unique=True)
    title = Column(String(500))
    content = Column(Text)
    page_type = Column(String(50))             # topic / insight / source / dream / index
    file_path = Column(String(500))
    source_chunks = Column(JSON, default=[])   # IDs of chunks used
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    git_sha = Column(String(64))               # Karpathy-style freshness tracking
    embedding = Column(Vector(768))

class DreamLog(Base):
    """سجل عمليات الـ Dreaming الليلية"""
    __tablename__ = "dream_logs"
    
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    conversations_processed = Column(Integer, default=0)
    pages_created = Column(Integer, default=0)
    pages_updated = Column(Integer, default=0)
    insights_found = Column(Integer, default=0)
    status = Column(String(20), default="running")  # running / done / failed
    summary = Column(Text)


class IngestJob(Base):
    """جدول تتبع مهام الاستيراد — يحل محل _jobs في الذاكرة"""
    __tablename__ = "ingest_jobs"

    id = Column(String(12), primary_key=True)       # job_id hex
    status = Column(String(20), default="running")  # running / done / error
    filename = Column(String(500))
    saved = Column(Integer, default=0)
    skipped = Column(Integer, default=0)
    total = Column(Integer, default=0)              # إجمالي المحادثات المكتشفة
    processed = Column(Integer, default=0)          # عدد المعالَجة حتى الآن (للـ progress bar)
    current_title = Column(String(500))             # عنوان المحادثة الجارية
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)

import os
import json
import hashlib
from typing import List, Optional
from sqlalchemy.orm import Session
from config.database import SessionLocal
from config.models import Conversation, Message, Chunk, SourceType
from ingestion.base_parser import ParsedConversation
from ingestion.chatgpt_parser import ChatGPTParser
from ingestion.claude_parser import ClaudeParser
from processing.embedder import Embedder
from rich.console import Console
from rich.progress import track

console = Console()

class IngestionService:
    """Main service to ingest conversation files into the database"""

    def __init__(self):
        self.parsers = [ChatGPTParser(), ClaudeParser()]
        self.embedder = Embedder()

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a single file (JSON or ZIP)"""
        console.print(f"\n📥 Ingesting: [cyan]{file_path}[/cyan]")

        # Auto-detect parser
        parser = self._detect_parser(file_path)
        if not parser:
            return {"error": f"Cannot detect format for: {file_path}"}

        # Parse conversations
        conversations = parser.parse(file_path)
        console.print(f"   Found [green]{len(conversations)}[/green] conversations")

        # Save to database
        saved = 0
        skipped = 0
        db = SessionLocal()
        try:
            for conv in track(conversations, description="   Processing..."):
                result = self._save_conversation(db, conv)
                if result == "saved":
                    saved += 1
                else:
                    skipped += 1
            db.commit()
        finally:
            db.close()

        console.print(f"   ✅ Saved: [green]{saved}[/green]  Skipped (duplicate): [yellow]{skipped}[/yellow]")
        return {"saved": saved, "skipped": skipped, "total": len(conversations)}

    def ingest_directory(self, dir_path: str) -> dict:
        """Ingest all JSON/ZIP files in a directory"""
        total = {"saved": 0, "skipped": 0, "total": 0}
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(('.json', '.zip'))
        ]
        for file_path in files:
            result = self.ingest_file(file_path)
            if "error" not in result:
                total["saved"] += result["saved"]
                total["skipped"] += result["skipped"]
                total["total"] += result["total"]
        return total

    def _detect_parser(self, file_path: str):
        """Auto-detect which parser to use"""
        data = None
        if file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                return None

        for parser in self.parsers:
            if parser.can_parse(file_path, data):
                return parser
        return None

    def _save_conversation(self, db: Session, parsed: ParsedConversation) -> str:
        """Save a parsed conversation to DB, skip if duplicate"""
        # Check for duplicate by original_id
        existing = db.query(Conversation).filter(
            Conversation.original_id == parsed.original_id,
            Conversation.source == parsed.source
        ).first()
        if existing:
            return "skipped"

        # Create conversation record
        conv = Conversation(
            source=parsed.source,
            title=parsed.title,
            original_id=parsed.original_id,
            created_at=parsed.created_at,
            raw_file=parsed.raw_file,
            metadata_=parsed.metadata
        )
        db.add(conv)
        db.flush()  # get conv.id

        # Save messages
        for msg in parsed.messages:
            message = Message(
                conversation_id=conv.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
                order_index=msg.order_index
            )
            db.add(message)

        # Create chunks + embeddings
        self._create_chunks(db, conv, parsed)

        return "saved"

    def _create_chunks(self, db: Session, conv: Conversation, parsed: ParsedConversation):
        """Split conversation into chunks and embed them"""
        # Pair user questions with assistant answers
        messages = parsed.messages
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == 'user' and i + 1 < len(messages) and messages[i+1].role == 'assistant':
                # Q&A pair chunk
                q = msg.content
                a = messages[i+1].content
                content = f"Q: {q}\n\nA: {a}"
                chunk_type = "qa_pair"
                i += 2
            else:
                content = msg.content
                chunk_type = msg.role
                i += 1

            # Skip very short chunks
            if len(content) < 50:
                continue

            # Truncate very long chunks
            if len(content) > 3000:
                content = content[:3000]

            # Generate embedding
            embedding = self.embedder.embed(content)

            chunk = Chunk(
                conversation_id=conv.id,
                content=content,
                embedding=embedding,
                chunk_type=chunk_type,
                topics=[]
            )
            db.add(chunk)

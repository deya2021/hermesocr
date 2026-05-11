import os
import json
import zipfile
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
        self.chatgpt_parser = ChatGPTParser()
        self.claude_parser  = ClaudeParser()
        self.embedder       = Embedder()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> dict:
        console.print(f"\n📥 Ingesting: [cyan]{file_path}[/cyan]")

        parser = self._detect_parser(file_path)
        if not parser:
            return {"error": f"لا يمكن التعرف على صيغة الملف: {os.path.basename(file_path)}"}

        conversations = parser.parse(file_path)
        console.print(f"   وجدنا [green]{len(conversations)}[/green] محادثة")

        saved = skipped = 0
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

        console.print(
            f"   ✅ Saved: [green]{saved}[/green]  "
            f"Skipped (duplicate): [yellow]{skipped}[/yellow]"
        )
        return {"saved": saved, "skipped": skipped, "total": len(conversations)}

    def ingest_directory(self, dir_path: str) -> dict:
        total = {"saved": 0, "skipped": 0, "total": 0}
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(('.json', '.zip'))
        ]
        for fp in files:
            result = self.ingest_file(fp)
            if "error" not in result:
                total["saved"]   += result["saved"]
                total["skipped"] += result["skipped"]
                total["total"]   += result["total"]
        return total

    # ──────────────────────────────────────────────────────────
    # Parser detection  ← FIXED
    # ──────────────────────────────────────────────────────────

    def _detect_parser(self, file_path: str):
        """
        Smart detection — handles ZIP and JSON correctly.

        ZIP heuristic:
          • Claude ZIP:   contains 'users.json' OR 'memories.json'
          • ChatGPT ZIP:  contains 'conversations.json' (no users.json)

        JSON heuristic:
          • ChatGPT JSON: list where first item has 'mapping' key
          • Claude JSON:  list where first item has 'chat_messages' or 'uuid'+'name'
        """
        ext = os.path.splitext(file_path)[1].lower()

        # ── ZIP ──────────────────────────────────────────────────
        if ext == '.zip':
            try:
                with zipfile.ZipFile(file_path) as z:
                    names = z.namelist()
                # Claude export always has users.json and memories.json
                if 'users.json' in names or 'memories.json' in names:
                    console.print("   [dim]Detected: Claude ZIP[/dim]")
                    return self.claude_parser
                # ChatGPT export: conversations.json at root, no users.json
                if 'conversations.json' in names:
                    console.print("   [dim]Detected: ChatGPT ZIP[/dim]")
                    return self.chatgpt_parser
            except Exception as e:
                console.print(f"   [red]ZIP read error: {e}[/red]")
            return None

        # ── JSON ─────────────────────────────────────────────────
        if ext == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                return None

            if not isinstance(data, list) or not data:
                return None

            first = data[0]
            if not isinstance(first, dict):
                return None

            # ChatGPT: has 'mapping' (linked-list of nodes)
            if 'mapping' in first:
                console.print("   [dim]Detected: ChatGPT JSON[/dim]")
                return self.chatgpt_parser

            # Claude: has 'chat_messages' field
            if 'chat_messages' in first:
                console.print("   [dim]Detected: Claude JSON[/dim]")
                return self.claude_parser

            # Claude fallback: uuid + name (no mapping)
            if 'uuid' in first and 'name' in first and 'mapping' not in first:
                console.print("   [dim]Detected: Claude JSON (uuid format)[/dim]")
                return self.claude_parser

        return None

    # ──────────────────────────────────────────────────────────
    # Save to DB
    # ──────────────────────────────────────────────────────────

    def _save_conversation(self, db: Session, parsed: ParsedConversation) -> str:
        # Skip duplicates
        existing = db.query(Conversation).filter(
            Conversation.original_id == parsed.original_id,
            Conversation.source      == parsed.source
        ).first()
        if existing:
            return "skipped"

        conv = Conversation(
            source      = parsed.source,
            title       = parsed.title,
            original_id = parsed.original_id,
            created_at  = parsed.created_at,
            raw_file    = parsed.raw_file,
            metadata_   = parsed.metadata
        )
        db.add(conv)
        db.flush()

        for msg in parsed.messages:
            db.add(Message(
                conversation_id = conv.id,
                role            = msg.role,
                content         = msg.content,
                created_at      = msg.created_at,
                order_index     = msg.order_index
            ))

        self._create_chunks(db, conv, parsed)
        return "saved"

    def _create_chunks(self, db: Session, conv: Conversation, parsed: ParsedConversation):
        """
        Create Q&A pair chunks (or single-message chunks) with embeddings.
        Embeddings are generated one-by-one (Ollama is a local serial service).
        """
        messages = parsed.messages
        i = 0
        while i < len(messages):
            msg = messages[i]

            # Q&A pair
            if (msg.role == 'user'
                    and i + 1 < len(messages)
                    and messages[i + 1].role == 'assistant'):
                q = msg.content[:1500]
                a = messages[i + 1].content[:1500]
                content    = f"Q: {q}\n\nA: {a}"
                chunk_type = "qa_pair"
                i += 2
            else:
                content    = msg.content[:3000]
                chunk_type = msg.role
                i += 1

            if len(content) < 50:
                continue

            embedding = self.embedder.embed(content)

            db.add(Chunk(
                conversation_id = conv.id,
                content         = content,
                embedding       = embedding,
                chunk_type      = chunk_type,
                topics          = []
            ))

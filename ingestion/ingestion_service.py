import os
import json
import zipfile
from typing import List, Optional, Callable
from sqlalchemy.orm import Session
from config.database import SessionLocal
from config.models import Conversation, Message, Chunk, SourceType, IngestJob
from ingestion.base_parser import ParsedConversation
from ingestion.chatgpt_parser import ChatGPTParser
from ingestion.claude_parser import ClaudeParser
from ingestion.gemini_parser import GeminiParser
from processing.embedder import Embedder
from rich.console import Console

console = Console()

# عدد المحادثات بين كل commit لتجنب transactions طويلة
BATCH_COMMIT_SIZE = 50


class IngestionService:
    """Main service to ingest conversation files into the database"""

    def __init__(self):
        self.chatgpt_parser = ChatGPTParser()
        self.claude_parser  = ClaudeParser()
        self.gemini_parser  = GeminiParser()
        self.embedder       = Embedder()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def ingest_file(self, file_path: str, job_id: str = None) -> dict:
        console.print(f"\n📥 Ingesting: [cyan]{file_path}[/cyan]")

        parser = self._detect_parser(file_path)
        if not parser:
            return {"error": f"لا يمكن التعرف على صيغة الملف: {os.path.basename(file_path)}"}

        # ── للملفات الكبيرة: streaming مع تحديث التقدم ─────────
        file_size = os.path.getsize(file_path)
        is_large  = file_size > 20 * 1024 * 1024

        if is_large:
            console.print(f"   📦 ملف كبير ({file_size//1024//1024}MB) — streaming mode")
            return self._ingest_streaming(file_path, parser, job_id)
        else:
            conversations = parser.parse(file_path)
            console.print(f"   وجدنا [green]{len(conversations)}[/green] محادثة")
            return self._save_all(conversations, job_id)

    def _ingest_streaming(self, file_path: str, parser, job_id: str) -> dict:
        """
        للملفات الكبيرة: نقرأ المحادثات واحدة واحدة بدون تحميل الكل في الذاكرة،
        ونُحدِّث قاعدة البيانات كل BATCH_COMMIT_SIZE محادثة.
        """
        saved   = 0
        skipped = 0
        total   = 0
        db      = SessionLocal()

        try:
            # generator — لا يُحمِّل كل شيء دفعة واحدة
            conv_stream = parser.parse(file_path)

            batch = []
            for conv in conv_stream:
                total += 1
                batch.append(conv)

                # حفظ كل BATCH_COMMIT_SIZE محادثة
                if len(batch) >= BATCH_COMMIT_SIZE:
                    s, sk = self._save_batch(db, batch, job_id, total)
                    saved   += s
                    skipped += sk
                    batch    = []
                    console.print(
                        f"   💾 Batch commit — total so far: {total} "
                        f"(saved:{saved} skipped:{skipped})"
                    )

            # الدفعة الأخيرة
            if batch:
                s, sk = self._save_batch(db, batch, job_id, total)
                saved   += s
                skipped += sk

        finally:
            db.close()

        console.print(
            f"   ✅ Saved: [green]{saved}[/green]  "
            f"Skipped (duplicate): [yellow]{skipped}[/yellow]  "
            f"Total: {total}"
        )
        return {"saved": saved, "skipped": skipped, "total": total}

    def _save_batch(
        self, db: Session, batch: list, job_id: str, total_so_far: int
    ):
        """حفظ دفعة من المحادثات وتحديث حالة الـ job."""
        saved = skipped = 0
        for conv in batch:
            result = self._save_conversation(db, conv)
            if result == "saved":
                saved += 1
            else:
                skipped += 1

        db.commit()

        # تحديث تقدم الـ job في DB
        if job_id:
            self._update_job_progress(
                job_id,
                processed = total_so_far,
                saved     = saved,
                skipped   = skipped,
                title     = batch[-1].title if batch else None,
            )

        return saved, skipped

    def _save_all(self, conversations: list, job_id: str) -> dict:
        """للملفات الصغيرة: الطريقة الكلاسيكية."""
        saved = skipped = 0
        db = SessionLocal()
        try:
            for i, conv in enumerate(conversations):
                result = self._save_conversation(db, conv)
                if result == "saved":
                    saved += 1
                else:
                    skipped += 1
                # commit كل BATCH_COMMIT_SIZE
                if (i + 1) % BATCH_COMMIT_SIZE == 0:
                    db.commit()
                    if job_id:
                        self._update_job_progress(
                            job_id,
                            processed = i + 1,
                            saved     = saved,
                            skipped   = skipped,
                            title     = conv.title,
                        )
            db.commit()
        finally:
            db.close()

        return {"saved": saved, "skipped": skipped, "total": len(conversations)}

    def _update_job_progress(
        self, job_id: str, processed: int, saved: int, skipped: int, title: str = None
    ):
        """تحديث حقول التقدم في IngestJob بدون فتح session جديدة."""
        try:
            db2 = SessionLocal()
            job = db2.query(IngestJob).filter(IngestJob.id == job_id).first()
            if job:
                job.processed      = processed
                job.saved          = (job.saved or 0) + saved
                job.skipped        = (job.skipped or 0) + skipped
                if title:
                    job.current_title = title[:500]
                db2.commit()
            db2.close()
        except Exception as e:
            console.print(f"   [yellow]⚠ job progress update failed: {e}[/yellow]")

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
    # Parser detection
    # ──────────────────────────────────────────────────────────

    def _detect_parser(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.zip':
            try:
                with zipfile.ZipFile(file_path) as z:
                    names = z.namelist()
                if 'users.json' in names or 'memories.json' in names:
                    console.print("   [dim]Detected: Claude ZIP[/dim]")
                    return self.claude_parser
                name_str = ' '.join(n.lower() for n in names)
                if ('gemini' in name_str or 'google ai' in name_str
                        or any(n.startswith('conversations/') for n in names)):
                    console.print("   [dim]Detected: Gemini ZIP[/dim]")
                    return self.gemini_parser
                if 'conversations.json' in names:
                    console.print("   [dim]Detected: ChatGPT ZIP[/dim]")
                    return self.chatgpt_parser
            except Exception as e:
                console.print(f"   [red]ZIP read error: {e}[/red]")
            return None

        if ext == '.json':
            # للملفات الكبيرة: نقرأ فقط أول 4KB لكشف النوع
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    head = f.read(4096)
            except Exception:
                return None

            # ChatGPT: يحتوي على "mapping" في العناصر الأولى
            if '"mapping"' in head:
                console.print("   [dim]Detected: ChatGPT JSON (large file)[/dim]")
                return self.chatgpt_parser

            # نحاول قراءة أول عنصر فقط
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                # إذا فشلت قراءة الكل، نجرب الـ head
                if '"mapping"' in head:
                    return self.chatgpt_parser
                return None

            # Format C — Gemini Takeout
            if isinstance(data, dict) and "conversations" in data:
                if self.gemini_parser._looks_like_gemini(data):
                    console.print("   [dim]Detected: Gemini JSON (Takeout)[/dim]")
                    return self.gemini_parser

            if not isinstance(data, list) or not data:
                return None

            first = data[0]
            if not isinstance(first, dict):
                return None

            if 'mapping' in first:
                console.print("   [dim]Detected: ChatGPT JSON[/dim]")
                return self.chatgpt_parser
            if 'chat_messages' in first:
                console.print("   [dim]Detected: Claude JSON[/dim]")
                return self.claude_parser
            if 'uuid' in first and 'name' in first and 'createTime' not in first:
                console.print("   [dim]Detected: Claude JSON (uuid)[/dim]")
                return self.claude_parser
            if self.gemini_parser._looks_like_gemini(data):
                console.print("   [dim]Detected: Gemini JSON[/dim]")
                return self.gemini_parser

        return None

    # ──────────────────────────────────────────────────────────
    # Save to DB
    # ──────────────────────────────────────────────────────────

    def _save_conversation(self, db: Session, parsed: ParsedConversation) -> str:
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
            metadata_   = parsed.metadata,
            is_digested = False,
        )
        db.add(conv)
        db.flush()

        for msg in parsed.messages:
            db.add(Message(
                conversation_id = conv.id,
                role            = msg.role,
                content         = msg.content,
                created_at      = msg.created_at,
                order_index     = msg.order_index,
            ))

        self._create_chunks(db, conv, parsed)
        return "saved"

    def _create_chunks(self, db: Session, conv: Conversation, parsed: ParsedConversation):
        messages = parsed.messages
        i = 0
        while i < len(messages):
            msg = messages[i]
            if (msg.role == 'user'
                    and i + 1 < len(messages)
                    and messages[i + 1].role == 'assistant'):
                q          = msg.content[:1500]
                a          = messages[i + 1].content[:1500]
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
                topics          = [],
            ))

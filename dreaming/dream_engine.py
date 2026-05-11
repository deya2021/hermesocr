import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from config.database import SessionLocal
from config.models import Conversation, Chunk, WikiPage, DreamLog
from config.settings import settings
from processing.llm_agent import LLMAgent
from processing.embedder import Embedder
from rich.console import Console

console = Console()


class DreamEngine:
    """
    Memory consolidation engine — Karpathy-style LLM Wiki.

    Phases:
    1. DIGEST   — summarize conversations → sources/ wiki pages
    2. CLUSTER  — collect topics from chunks
    3. WRITE    — generate topic pages (limited per run)
    4. CONNECT  — find cross-topic insights → dreams/ page
    5. COMPRESS — skip on first run (no old pages yet)
    6. INDEX    — rebuild index.md + overview.md

    Optimized for CPU-only server (no GPU):
    - Uses llama3.2:1b (fast) instead of hermes3:8b (slow)
    - Short prompts, low num_predict
    - Processes max 5 convs + 5 topics per run to stay fast
    - Each phase logs progress so failures are visible
    """

    # Limits per dreaming run to stay under ~20 min on CPU
    MAX_CONVS_PER_RUN   = 5
    MAX_TOPICS_PER_RUN  = 5

    def __init__(self):
        self.agent     = LLMAgent()
        self.embedder  = Embedder()
        self.wiki_path = settings.wiki_path

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def run(self, full: bool = False) -> DreamLog:
        db  = SessionLocal()
        log = DreamLog(started_at=datetime.utcnow(), status="running")
        db.add(log)
        db.commit()
        db.refresh(log)

        try:
            console.print("\n[bold magenta]🌙 DREAMING STARTED[/bold magenta]")

            # ── Phase 1: DIGEST ──────────────────────────────
            console.print("\n[cyan]Phase 1/6: DIGEST[/cyan]")
            convs = self._get_unprocessed(db, full)
            console.print(f"  {len(convs)} conversations to digest")
            for i, conv in enumerate(convs):
                console.print(f"  [{i+1}/{len(convs)}] {conv.title[:60]}")
                try:
                    result = self._digest_one(db, conv)
                    log.conversations_processed += 1
                    if result == "created":
                        log.pages_created += 1
                    else:
                        log.pages_updated += 1
                    db.commit()
                except Exception as e:
                    console.print(f"  [red]⚠ digest error: {e}[/red]")
                    db.rollback()

            # ── Phase 2-3: CLUSTER + WRITE ───────────────────
            console.print("\n[cyan]Phase 2-3/6: CLUSTER + WRITE[/cyan]")
            topics = self._get_top_topics(db)
            console.print(f"  {len(topics)} topics found, writing top {min(len(topics), self.MAX_TOPICS_PER_RUN)}")
            for i, topic in enumerate(topics[:self.MAX_TOPICS_PER_RUN]):
                console.print(f"  [{i+1}] topic: {topic}")
                try:
                    result = self._write_topic_page(db, topic)
                    if result == "created":
                        log.pages_created += 1
                    elif result == "updated":
                        log.pages_updated += 1
                    db.commit()
                except Exception as e:
                    console.print(f"  [red]⚠ topic error: {e}[/red]")
                    db.rollback()

            # ── Phase 4: CONNECT ─────────────────────────────
            console.print("\n[cyan]Phase 4/6: CONNECT[/cyan]")
            try:
                result = self._find_connections(db)
                if result in ("created", "updated"):
                    log.insights_found += 1
                    if result == "created":
                        log.pages_created += 1
                db.commit()
            except Exception as e:
                console.print(f"  [red]⚠ connect error: {e}[/red]")
                db.rollback()

            # ── Phase 5: COMPRESS ────────────────────────────
            console.print("\n[cyan]Phase 5/6: COMPRESS[/cyan]")
            try:
                updated = self._compress_pages(db)
                log.pages_updated += updated
                db.commit()
            except Exception as e:
                console.print(f"  [red]⚠ compress error: {e}[/red]")
                db.rollback()

            # ── Phase 6: INDEX ───────────────────────────────
            console.print("\n[cyan]Phase 6/6: INDEX[/cyan]")
            try:
                self._rebuild_index(db)
                log.pages_updated += 2
                db.commit()
            except Exception as e:
                console.print(f"  [red]⚠ index error: {e}[/red]")
                db.rollback()

            log.status      = "done"
            log.finished_at = datetime.utcnow()
            elapsed = (log.finished_at - log.started_at).seconds // 60
            log.summary = (
                f"Completed in {elapsed} min. "
                f"Convs: {log.conversations_processed}, "
                f"Created: {log.pages_created}, "
                f"Updated: {log.pages_updated}, "
                f"Insights: {log.insights_found}"
            )
            db.commit()

            console.print(f"\n[bold green]✅ DREAMING DONE in {elapsed} min[/bold green]")
            console.print(f"   Convs processed : {log.conversations_processed}")
            console.print(f"   Pages created   : {log.pages_created}")
            console.print(f"   Pages updated   : {log.pages_updated}")
            console.print(f"   Insights        : {log.insights_found}")

        except Exception as e:
            log.status      = "failed"
            log.summary     = str(e)
            log.finished_at = datetime.utcnow()
            try:
                db.commit()
            except Exception:
                db.rollback()
            console.print(f"\n[bold red]❌ DREAMING FAILED: {e}[/bold red]")
        finally:
            db.close()

        return log

    # ──────────────────────────────────────────────────────────
    # Phase 1 helpers
    # ──────────────────────────────────────────────────────────

    def _get_unprocessed(self, db: Session, full: bool) -> List[Conversation]:
        """Return conversations that have no source wiki page yet."""
        if full:
            return (db.query(Conversation)
                      .order_by(Conversation.created_at)
                      .limit(self.MAX_CONVS_PER_RUN)
                      .all())

        # Find conv IDs already digested (stored in source_chunks of source pages)
        done_ids: set = set()
        source_pages = db.query(WikiPage.source_chunks).filter(
            WikiPage.page_type == "source"
        ).all()
        for row in source_pages:
            if row[0]:
                done_ids.update(row[0])

        q = db.query(Conversation).order_by(Conversation.created_at)
        if done_ids:
            q = q.filter(~Conversation.id.in_(done_ids))
        return q.limit(self.MAX_CONVS_PER_RUN).all()

    def _digest_one(self, db: Session, conv: Conversation) -> str:
        """Summarize one conversation → source wiki page."""
        messages = sorted(conv.messages, key=lambda m: m.order_index)

        # Build a compact text — first 15 messages, 400 chars each
        msgs_text = "\n\n".join(
            f"{m.role.upper()}: {m.content[:400]}"
            for m in messages[:15]
        )

        # LLM: summarize
        content = self.agent.summarize_conversation(conv.title, msgs_text)

        # LLM: extract topics → attach to chunks
        topics = self.agent.extract_topics(msgs_text)
        console.print(f"    topics: {topics}")
        for chunk in conv.chunks:
            chunk.topics = topics

        slug = f"sources/{conv.source.value}_{conv.id}"
        return self._upsert_page(
            db       = db,
            slug     = slug,
            title    = f"{conv.source.value.title()}: {conv.title}",
            content  = content,
            page_type= "source",
            src_chunks = [c.id for c in conv.chunks],
            subpath  = f"sources/{conv.source.value}_{conv.id}.md",
        )

    # ──────────────────────────────────────────────────────────
    # Phase 2-3 helpers
    # ──────────────────────────────────────────────────────────

    def _get_top_topics(self, db: Session) -> List[str]:
        rows = db.query(Chunk.topics).filter(Chunk.topics.isnot(None)).all()
        counts: Dict[str, int] = {}
        for row in rows:
            for t in (row[0] or []):
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts, key=lambda t: -counts[t])

    def _write_topic_page(self, db: Session, topic: str) -> str:
        emb = self.embedder.embed(topic)
        rows = db.execute(text("""
            SELECT content,
                   1 - (embedding <=> CAST(:emb AS vector)) AS sim
            FROM   chunks
            WHERE  1 - (embedding <=> CAST(:emb AS vector)) > 0.4
            ORDER  BY sim DESC
            LIMIT  6
        """), {"emb": str(emb)}).fetchall()

        if not rows:
            console.print(f"    no chunks found for topic '{topic}', skipping")
            return "skipped"

        chunks = [r[0][:600] for r in rows]
        content = self.agent.generate_topic_page(topic, chunks)

        safe = topic.replace(" ", "_").replace("/", "-").lower()
        return self._upsert_page(
            db        = db,
            slug      = f"topics/{safe}",
            title     = f"Topic: {topic.title()}",
            content   = content,
            page_type = "topic",
            src_chunks= [],
            subpath   = f"topics/{safe}.md",
        )

    # ──────────────────────────────────────────────────────────
    # Phase 4: CONNECT
    # ──────────────────────────────────────────────────────────

    def _find_connections(self, db: Session) -> str:
        topics = self._get_top_topics(db)
        if len(topics) < 2:
            console.print("  not enough topics yet, skipping")
            return "skipped"

        pages = (db.query(WikiPage)
                   .filter(WikiPage.page_type == "source")
                   .order_by(WikiPage.updated_at.desc())
                   .limit(10)
                   .all())
        summaries = [p.content[:300] for p in pages]

        content  = self.agent.find_connections(topics, summaries)
        date_str = datetime.utcnow().strftime("%Y_%m_%d")
        return self._upsert_page(
            db        = db,
            slug      = f"dreams/connections_{date_str}",
            title     = f"Dream: Connections — {date_str}",
            content   = content,
            page_type = "dream",
            src_chunks= [],
            subpath   = f"dreams/connections_{date_str}.md",
        )

    # ──────────────────────────────────────────────────────────
    # Phase 5: COMPRESS
    # ──────────────────────────────────────────────────────────

    def _compress_pages(self, db: Session) -> int:
        """Merge yesterday's new chunks into existing topic pages."""
        cutoff     = datetime.utcnow() - timedelta(days=1)
        new_chunks = db.query(Chunk).filter(Chunk.created_at >= cutoff).all()
        if not new_chunks:
            console.print("  no new chunks, skipping compress")
            return 0

        # Group by topic
        by_topic: Dict[str, List[str]] = {}
        for ch in new_chunks:
            for t in (ch.topics or []):
                by_topic.setdefault(t, []).append(ch.content)

        updated = 0
        for topic, chunks in list(by_topic.items())[:3]:   # max 3 per run
            safe = topic.replace(" ", "_").replace("/", "-").lower()
            page = db.query(WikiPage).filter(WikiPage.slug == f"topics/{safe}").first()
            if page and page.content:
                new_content       = self.agent.compress_knowledge(page.content, chunks)
                page.content      = new_content
                page.updated_at   = datetime.utcnow()
                page.git_sha      = hashlib.sha256(new_content.encode()).hexdigest()[:16]
                self._write_md(page.file_path, new_content)
                updated += 1
        return updated

    # ──────────────────────────────────────────────────────────
    # Phase 6: INDEX
    # ──────────────────────────────────────────────────────────

    def _rebuild_index(self, db: Session):
        conv_count = db.query(func.count(Conversation.id)).scalar()
        topics     = self._get_top_topics(db)

        dates = db.query(
            func.min(Conversation.created_at),
            func.max(Conversation.created_at),
        ).first()
        date_range = (
            f"{dates[0].strftime('%Y-%m-%d') if dates[0] else '?'}"
            f" → "
            f"{dates[1].strftime('%Y-%m-%d') if dates[1] else '?'}"
        )

        # Overview page via LLM
        overview = self.agent.generate_overview(
            topics,
            {"conversations": conv_count, "topics": len(topics), "date_range": date_range},
        )
        self._upsert_page(db, "overview", "Knowledge Overview",
                          overview, "index", [], "overview.md")

        # Static index
        pages = db.query(WikiPage).order_by(WikiPage.updated_at.desc()).all()
        lines = [
            "# 📚 Memory Wiki Index\n",
            f"> Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n",
            f"\n**{conv_count}** محادثات · **{len(topics)}** مواضيع\n",
        ]
        for section, ptype in [("## 📖 Topics", "topic"),
                                ("## 🌙 Dreams", "dream"),
                                ("## 📂 Sources", "source")]:
            lines.append(f"\n{section}\n")
            for p in pages:
                if p.page_type == ptype:
                    lines.append(f"- [{p.title}]({p.file_path})")

        self._upsert_page(db, "index", "Index",
                          "\n".join(lines), "index", [], "index.md")

    # ──────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────

    def _upsert_page(self, db: Session, slug: str, title: str,
                     content: str, page_type: str,
                     src_chunks: list, subpath: str) -> str:
        sha       = hashlib.sha256(content.encode()).hexdigest()[:16]
        file_path = os.path.join(self.wiki_path, subpath)
        self._write_md(file_path, content)

        page = db.query(WikiPage).filter(WikiPage.slug == slug).first()
        if page:
            page.content    = content
            page.updated_at = datetime.utcnow()
            page.git_sha    = sha
            return "updated"
        else:
            db.add(WikiPage(
                slug          = slug,
                title         = title,
                content       = content,
                page_type     = page_type,
                file_path     = file_path,
                source_chunks = src_chunks,
                git_sha       = sha,
            ))
            return "created"

    def _write_md(self, file_path: str, content: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

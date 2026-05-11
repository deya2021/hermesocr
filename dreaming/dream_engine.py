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
    Nightly memory consolidation engine (Karpathy-style LLM Wiki + Dreaming).
    
    Phases:
    1. DIGEST   - Summarize new conversations → sources/ pages
    2. CLUSTER  - Group chunks by topic similarity
    3. WRITE    - Generate/update topic pages
    4. CONNECT  - Find cross-topic insights → dreams/ pages
    5. COMPRESS - Merge redundant knowledge
    6. INDEX    - Rebuild master index + overview
    """

    def __init__(self):
        self.agent = LLMAgent()
        self.embedder = Embedder()
        self.wiki_path = settings.wiki_path

    def run(self, full: bool = False) -> DreamLog:
        """Run the full dreaming cycle"""
        db = SessionLocal()
        log = DreamLog(started_at=datetime.utcnow(), status="running")
        db.add(log)
        db.commit()

        try:
            console.print("\n[bold magenta]🌙 DREAMING STARTED[/bold magenta]")
            
            # Phase 1: Digest new conversations
            console.print("\n[cyan]Phase 1: DIGEST[/cyan] — Summarizing new conversations...")
            new_convs = self._get_unprocessed_conversations(db, full)
            console.print(f"  Found {len(new_convs)} conversations to process")
            
            for conv in new_convs:
                self._digest_conversation(db, conv, log)

            # Phase 2 & 3: Cluster topics and write topic pages
            console.print("\n[cyan]Phase 2-3: CLUSTER + WRITE[/cyan] — Building topic pages...")
            topics = self._get_all_topics(db)
            for topic in topics[:20]:  # limit per run
                self._write_topic_page(db, topic, log)

            # Phase 4: Connect — find cross-topic insights
            console.print("\n[cyan]Phase 4: CONNECT[/cyan] — Finding connections...")
            self._find_connections(db, log)

            # Phase 5: Compress — merge redundant pages
            console.print("\n[cyan]Phase 5: COMPRESS[/cyan] — Compressing knowledge...")
            self._compress_old_pages(db, log)

            # Phase 6: Index — rebuild master pages
            console.print("\n[cyan]Phase 6: INDEX[/cyan] — Rebuilding index...")
            self._rebuild_index(db, log)

            log.status = "done"
            log.finished_at = datetime.utcnow()
            db.commit()
            
            console.print(f"\n[bold green]✅ DREAMING COMPLETE[/bold green]")
            console.print(f"  Conversations: {log.conversations_processed}")
            console.print(f"  Pages created: {log.pages_created}")
            console.print(f"  Pages updated: {log.pages_updated}")
            console.print(f"  Insights:      {log.insights_found}")

        except Exception as e:
            log.status = "failed"
            log.summary = str(e)
            log.finished_at = datetime.utcnow()
            db.commit()
            console.print(f"\n[bold red]❌ DREAMING FAILED: {e}[/bold red]")
            raise
        finally:
            db.close()

        return log

    # ─────────────────────────────────────────
    # Phase 1: DIGEST
    # ─────────────────────────────────────────
    def _get_unprocessed_conversations(self, db: Session, full: bool) -> List[Conversation]:
        query = db.query(Conversation)
        if not full:
            # Only conversations without a wiki source page
            processed_ids = db.query(WikiPage.source_chunks).filter(
                WikiPage.page_type == 'source'
            ).all()
            processed_conv_ids = set()
            for row in processed_ids:
                if row[0]:
                    for cid in row[0]:
                        processed_conv_ids.add(cid)
            if processed_conv_ids:
                query = query.filter(~Conversation.id.in_(processed_conv_ids))
        return query.order_by(Conversation.created_at).limit(50).all()

    def _digest_conversation(self, db: Session, conv: Conversation, log: DreamLog):
        """Create/update source wiki page for a conversation"""
        # Build messages text
        messages = sorted(conv.messages, key=lambda m: m.order_index)
        messages_text = "\n\n".join([
            f"**{m.role.upper()}**: {m.content[:500]}"
            for m in messages[:20]
        ])

        # Generate wiki page content via LLM
        content = self.agent.summarize_conversation(conv.title, messages_text)

        # Extract and save topics back to chunks
        topics = self.agent.extract_topics(messages_text)
        for chunk in conv.chunks:
            chunk.topics = topics
        
        # Save wiki page
        slug = f"sources/{conv.source.value}_{conv.id}"
        page = self._save_wiki_page(
            db=db,
            slug=slug,
            title=f"{conv.source.value.title()}: {conv.title}",
            content=content,
            page_type="source",
            source_chunks=[c.id for c in conv.chunks],
            file_subpath=f"sources/{conv.source.value}_{conv.id}.md"
        )

        db.commit()
        log.conversations_processed += 1
        if page == "created":
            log.pages_created += 1
        else:
            log.pages_updated += 1

    # ─────────────────────────────────────────
    # Phase 2-3: CLUSTER + WRITE
    # ─────────────────────────────────────────
    def _get_all_topics(self, db: Session) -> List[str]:
        """Get all unique topics from chunks"""
        chunks = db.query(Chunk.topics).filter(Chunk.topics != None).all()
        topic_count = {}
        for row in chunks:
            if row[0]:
                for t in row[0]:
                    topic_count[t] = topic_count.get(t, 0) + 1
        # Sort by frequency
        return sorted(topic_count.keys(), key=lambda t: -topic_count[t])

    def _write_topic_page(self, db: Session, topic: str, log: DreamLog):
        """Find relevant chunks and generate/update topic wiki page"""
        # Semantic search for this topic
        topic_embedding = self.embedder.embed(topic)
        
        # Get top chunks for this topic
        similar_chunks = db.execute(text("""
            SELECT content, 1 - (embedding <=> CAST(:emb AS vector)) as similarity
            FROM chunks
            WHERE 1 - (embedding <=> CAST(:emb AS vector)) > 0.5
            ORDER BY similarity DESC
            LIMIT 10
        """), {"emb": str(topic_embedding)}).fetchall()

        if not similar_chunks:
            return

        chunk_texts = [row[0] for row in similar_chunks]
        content = self.agent.generate_topic_page(topic, chunk_texts)

        slug = f"topics/{topic.replace(' ', '_').lower()}"
        result = self._save_wiki_page(
            db=db,
            slug=slug,
            title=f"Topic: {topic.title()}",
            content=content,
            page_type="topic",
            source_chunks=[],
            file_subpath=f"topics/{topic.replace(' ', '_').lower()}.md"
        )
        db.commit()
        if result == "created":
            log.pages_created += 1
        else:
            log.pages_updated += 1

    # ─────────────────────────────────────────
    # Phase 4: CONNECT
    # ─────────────────────────────────────────
    def _find_connections(self, db: Session, log: DreamLog):
        """Find cross-topic insights — the real 'dreaming' phase"""
        topics = self._get_all_topics(db)[:30]
        if len(topics) < 3:
            return

        # Get recent source page summaries
        recent_pages = db.query(WikiPage).filter(
            WikiPage.page_type == 'source'
        ).order_by(WikiPage.updated_at.desc()).limit(20).all()

        summaries = [p.content[:500] for p in recent_pages]

        connections_content = self.agent.find_connections(topics, summaries)
        
        date_str = datetime.utcnow().strftime("%Y_%m_%d")
        slug = f"dreams/connections_{date_str}"
        result = self._save_wiki_page(
            db=db,
            slug=slug,
            title=f"Dream: Connections — {date_str}",
            content=connections_content,
            page_type="dream",
            source_chunks=[],
            file_subpath=f"dreams/connections_{date_str}.md"
        )
        db.commit()
        log.insights_found += 1
        if result == "created":
            log.pages_created += 1

    # ─────────────────────────────────────────
    # Phase 5: COMPRESS
    # ─────────────────────────────────────────
    def _compress_old_pages(self, db: Session, log: DreamLog):
        """Update topic pages with recently added chunks"""
        cutoff = datetime.utcnow() - timedelta(days=1)
        new_chunks = db.query(Chunk).filter(
            Chunk.created_at >= cutoff
        ).all()

        if not new_chunks:
            return

        # Group new chunks by topic
        topic_new_chunks: Dict[str, List[str]] = {}
        for chunk in new_chunks:
            for topic in (chunk.topics or []):
                if topic not in topic_new_chunks:
                    topic_new_chunks[topic] = []
                topic_new_chunks[topic].append(chunk.content)

        for topic, chunks in topic_new_chunks.items():
            slug = f"topics/{topic.replace(' ', '_').lower()}"
            page = db.query(WikiPage).filter(WikiPage.slug == slug).first()
            if page and page.content:
                updated = self.agent.compress_knowledge(page.content, chunks)
                page.content = updated
                page.updated_at = datetime.utcnow()
                sha = hashlib.sha256(updated.encode()).hexdigest()[:16]
                page.git_sha = sha
                # Write to file
                self._write_md_file(page.file_path, updated)
                log.pages_updated += 1

        db.commit()

    # ─────────────────────────────────────────
    # Phase 6: INDEX
    # ─────────────────────────────────────────
    def _rebuild_index(self, db: Session, log: DreamLog):
        """Rebuild master index.md and overview.md"""
        # Stats
        conv_count = db.query(func.count(Conversation.id)).scalar()
        topics = self._get_all_topics(db)
        
        dates = db.query(
            func.min(Conversation.created_at),
            func.max(Conversation.created_at)
        ).first()
        date_range = f"{dates[0].strftime('%Y-%m-%d') if dates[0] else '?'} → {dates[1].strftime('%Y-%m-%d') if dates[1] else '?'}"

        stats = {
            "conversations": conv_count,
            "topics": len(topics),
            "date_range": date_range
        }

        # Overview page
        overview = self.agent.generate_overview(topics, stats)
        self._save_wiki_page(
            db=db,
            slug="overview",
            title="Knowledge Overview",
            content=overview,
            page_type="index",
            source_chunks=[],
            file_subpath="overview.md"
        )

        # Index page (static list)
        pages = db.query(WikiPage).order_by(WikiPage.updated_at.desc()).all()
        index_lines = [
            "# 📚 Memory Wiki Index\n",
            f"> Auto-generated | {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n",
            f"\n**{conv_count} conversations** · **{len(topics)} topics**\n",
            "\n## 📖 Topics\n"
        ]
        for p in pages:
            if p.page_type == 'topic':
                index_lines.append(f"- [{p.title}]({p.file_path})")
        index_lines.append("\n## 🌙 Dreams\n")
        for p in pages:
            if p.page_type == 'dream':
                index_lines.append(f"- [{p.title}]({p.file_path})")
        index_lines.append("\n## 📂 Sources\n")
        for p in pages:
            if p.page_type == 'source':
                index_lines.append(f"- [{p.title}]({p.file_path})")

        index_content = "\n".join(index_lines)
        self._save_wiki_page(
            db=db, slug="index", title="Index",
            content=index_content, page_type="index",
            source_chunks=[], file_subpath="index.md"
        )
        db.commit()
        log.pages_updated += 2

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────
    def _save_wiki_page(self, db: Session, slug: str, title: str,
                         content: str, page_type: str,
                         source_chunks: list, file_subpath: str) -> str:
        sha = hashlib.sha256(content.encode()).hexdigest()[:16]
        file_path = os.path.join(self.wiki_path, file_subpath)

        # Write markdown file
        self._write_md_file(file_path, content)

        # Upsert DB record
        page = db.query(WikiPage).filter(WikiPage.slug == slug).first()
        if page:
            page.content = content
            page.updated_at = datetime.utcnow()
            page.git_sha = sha
            return "updated"
        else:
            page = WikiPage(
                slug=slug, title=title, content=content,
                page_type=page_type, file_path=file_path,
                source_chunks=source_chunks, git_sha=sha
            )
            db.add(page)
            return "created"

    def _write_md_file(self, file_path: str, content: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

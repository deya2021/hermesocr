import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text, desc
from typing import List, Optional
from config.database import get_db
from config.models import Conversation, WikiPage, DreamLog, Chunk
from config.settings import settings

router = APIRouter()

# ─────────────────────────────────────────
# Stats
# ─────────────────────────────────────────
@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    conv_count    = db.query(func.count(Conversation.id)).scalar()
    page_count    = db.query(func.count(WikiPage.id)).scalar()
    chunk_count   = db.query(func.count(Chunk.id)).scalar()
    last_dream    = db.query(DreamLog).order_by(desc(DreamLog.started_at)).first()
    return {
        "conversations": conv_count,
        "wiki_pages":    page_count,
        "chunks":        chunk_count,
        "last_dream":    last_dream.started_at.isoformat() if last_dream else None,
        "last_dream_status": last_dream.status if last_dream else None,
    }

# ─────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────
@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a conversation export file"""
    allowed = {'.json', '.zip'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type {ext} not supported. Use .json or .zip")

    # Save uploaded file
    dest = os.path.join(settings.uploads_path, file.filename)
    os.makedirs(settings.uploads_path, exist_ok=True)
    with open(dest, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    from ingestion.ingestion_service import IngestionService
    svc = IngestionService()
    result = svc.ingest_file(dest)
    return result

# ─────────────────────────────────────────
# Search
# ─────────────────────────────────────────
@router.get("/search")
def search(
    q: str = Query(..., min_length=2),
    mode: str = Query("semantic", enum=["semantic", "keyword"]),
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db)
):
    """Search across all conversations and wiki pages"""
    if mode == "semantic":
        from processing.embedder import Embedder
        emb = Embedder().embed(q)
        rows = db.execute(text("""
            SELECT c.id, c.content, c.chunk_type,
                   cv.title, cv.source,
                   1 - (c.embedding <=> CAST(:emb AS vector)) AS score
            FROM chunks c
            JOIN conversations cv ON cv.id = c.conversation_id
            WHERE 1 - (c.embedding <=> CAST(:emb AS vector)) > 0.3
            ORDER BY score DESC
            LIMIT :lim
        """), {"emb": str(emb), "lim": limit}).fetchall()

        return [{"chunk_id": r[0], "content": r[1][:300],
                 "type": r[2], "conversation": r[3],
                 "source": r[4], "score": round(r[5], 3)} for r in rows]
    else:
        # Keyword search
        rows = db.query(Chunk).filter(
            Chunk.content.ilike(f"%{q}%")
        ).limit(limit).all()
        return [{"chunk_id": r.id, "content": r.content[:300],
                 "type": r.chunk_type} for r in rows]

# ─────────────────────────────────────────
# Wiki Pages
# ─────────────────────────────────────────
@router.get("/wiki")
def list_wiki_pages(
    page_type: Optional[str] = None,
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    q = db.query(WikiPage)
    if page_type:
        q = q.filter(WikiPage.page_type == page_type)
    pages = q.order_by(desc(WikiPage.updated_at)).limit(limit).all()
    return [{"id": p.id, "slug": p.slug, "title": p.title,
             "type": p.page_type, "updated_at": p.updated_at.isoformat()} for p in pages]

@router.get("/wiki/{slug:path}")
def get_wiki_page(slug: str, db: Session = Depends(get_db)):
    page = db.query(WikiPage).filter(WikiPage.slug == slug).first()
    if not page:
        raise HTTPException(404, "Page not found")
    return {"slug": page.slug, "title": page.title,
            "content": page.content, "type": page.page_type,
            "updated_at": page.updated_at.isoformat(), "sha": page.git_sha}

# ─────────────────────────────────────────
# Conversations
# ─────────────────────────────────────────
@router.get("/conversations")
def list_conversations(
    source: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    q = db.query(Conversation)
    if source:
        q = q.filter(Conversation.source == source)
    total = q.count()
    items = q.order_by(desc(Conversation.created_at)).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [{"id": c.id, "title": c.title, "source": c.source,
                   "created_at": c.created_at.isoformat() if c.created_at else None,
                   "messages_count": len(c.messages)} for c in items]
    }

@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    messages = sorted(conv.messages, key=lambda m: m.order_index)
    return {
        "id": conv.id, "title": conv.title, "source": conv.source,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "messages": [{"role": m.role, "content": m.content} for m in messages]
    }

# ─────────────────────────────────────────
# Dreaming
# ─────────────────────────────────────────
@router.post("/dream")
def trigger_dream(full: bool = False):
    """Manually trigger a dreaming cycle"""
    import threading
    from dreaming.dream_engine import DreamEngine
    def run():
        DreamEngine().run(full=full)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return {"status": "started", "message": "Dreaming cycle started in background"}

@router.get("/dreams")
def list_dreams(db: Session = Depends(get_db)):
    logs = db.query(DreamLog).order_by(desc(DreamLog.started_at)).limit(20).all()
    return [{"id": l.id, "started_at": l.started_at.isoformat(),
             "status": l.status, "conversations": l.conversations_processed,
             "pages_created": l.pages_created, "pages_updated": l.pages_updated,
             "insights": l.insights_found} for l in logs]

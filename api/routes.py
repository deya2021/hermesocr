import os
import shutil
import threading
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text, desc
from typing import List, Optional
from config.database import get_db, SessionLocal
from config.models import Conversation, WikiPage, DreamLog, Chunk, IngestJob
from config.settings import settings

router = APIRouter()

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # 500 MB

# ─────────────────────────────────────────
# Ingestion background worker
# ─────────────────────────────────────────
def _run_ingest(job_id: str, file_path: str, original_filename: str):
    """Run ingestion in a background thread, persist status to DB, then delete the file."""
    db = SessionLocal()
    try:
        from ingestion.ingestion_service import IngestionService
        svc = IngestionService()
        result = svc.ingest_file(file_path, job_id=job_id)

        job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
        if job:
            job.status      = "error" if result.get("error") else "done"
            job.saved       = result.get("saved", 0)
            job.skipped     = result.get("skipped", 0)
            job.total       = result.get("total", 0)
            job.error       = result.get("error")
            job.finished_at = datetime.utcnow()
            db.commit()
    except Exception as e:
        job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
        if job:
            job.status      = "error"
            job.error       = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
        # حذف الملف بعد الانتهاء سواء نجح أم فشل
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


# ─────────────────────────────────────────
# Stats
# ─────────────────────────────────────────
@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    conv_count  = db.query(func.count(Conversation.id)).scalar()
    page_count  = db.query(func.count(WikiPage.id)).scalar()
    chunk_count = db.query(func.count(Chunk.id)).scalar()
    last_dream  = db.query(DreamLog).order_by(desc(DreamLog.started_at)).first()
    # هل يوجد dream جارٍ الآن؟
    running_dream = db.query(DreamLog).filter(DreamLog.status == "running").first()
    return {
        "conversations":     conv_count,
        "wiki_pages":        page_count,
        "chunks":            chunk_count,
        "last_dream":        last_dream.started_at.isoformat() if last_dream else None,
        "last_dream_status": last_dream.status if last_dream else None,
        "dream_running":     running_dream is not None,
    }


# ─────────────────────────────────────────
# Ingestion  (non-blocking)
# ─────────────────────────────────────────
@router.post("/ingest")
async def ingest_file(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a conversation export file.
    Returns immediately with a job_id.
    Poll  GET /api/ingest/status/{job_id}  to check progress.
    """
    # ── التحقق من نوع الملف ──────────────────────────────────
    allowed = {'.json', '.zip'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"نوع الملف {ext} غير مدعوم. استخدم .json أو .zip")

    # ── التحقق من حجم الملف ─────────────────────────────────
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"حجم الملف يتجاوز الحد الأقصى المسموح (500 MB)")

    # ── حفظ الملف ────────────────────────────────────────────
    os.makedirs(settings.uploads_path, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = os.path.join(settings.uploads_path, safe_name)

    total_written = 0
    with open(dest, 'wb') as f:
        while True:
            chunk = await file.read(1024 * 1024)   # قراءة 1MB في كل مرة
            if not chunk:
                break
            total_written += len(chunk)
            if total_written > MAX_UPLOAD_BYTES:
                f.close()
                os.remove(dest)
                raise HTTPException(413, "حجم الملف يتجاوز الحد الأقصى المسموح (500 MB)")
            f.write(chunk)

    # ── إنشاء سجل في DB ──────────────────────────────────────
    job_id = uuid.uuid4().hex[:12]
    job    = IngestJob(
        id       = job_id,
        status   = "running",
        filename = file.filename,
    )
    db.add(job)
    db.commit()

    # ── تشغيل الخيط الخلفي ───────────────────────────────────
    t = threading.Thread(target=_run_ingest, args=(job_id, dest, file.filename), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "running", "filename": file.filename}


@router.get("/ingest/status/{job_id}")
def ingest_status(job_id: str, db: Session = Depends(get_db)):
    """Poll ingestion job status (persistent — survives server restarts)"""
    job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    # حساب النسبة المئوية الحقيقية للتقدم
    pct = 0
    if job.total and job.total > 0:
        pct = min(99, int((job.processed or 0) / job.total * 100))
    elif job.status == "done":
        pct = 100

    return {
        "job_id":        job.id,
        "status":        job.status,
        "saved":         job.saved,
        "skipped":       job.skipped,
        "total":         job.total,
        "processed":     job.processed or 0,
        "current_title": job.current_title,
        "pct":           pct,
        "error":         job.error,
    }


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
    if mode == "semantic":
        try:
            from processing.embedder import Embedder
            emb = Embedder().embed(q)
        except Exception as e:
            raise HTTPException(503, f"تعذّر الاتصال بـ Ollama للبحث الدلالي: {e}")

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
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    q = db.query(WikiPage)
    if page_type:
        q = q.filter(WikiPage.page_type == page_type)
    total = q.count()
    pages = q.order_by(desc(WikiPage.updated_at)).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [{"id": p.id, "slug": p.slug, "title": p.title,
                   "type": p.page_type, "updated_at": p.updated_at.isoformat()} for p in pages]
    }


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
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    q = db.query(Conversation)
    if source:
        q = q.filter(Conversation.source == source)
    total = q.count()
    items = q.order_by(desc(Conversation.created_at)).offset(offset).limit(limit).all()
    return {
        "total": total,
        "has_more": (offset + limit) < total,
        "items": [{"id": c.id, "title": c.title, "source": c.source,
                   "created_at": c.created_at.isoformat() if c.created_at else None,
                   "messages_count": len(c.messages),
                   "is_digested": c.is_digested} for c in items]
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
        "is_digested": conv.is_digested,
        "messages": [{"role": m.role, "content": m.content} for m in messages]
    }


# ─────────────────────────────────────────
# Dreaming
# ─────────────────────────────────────────
@router.post("/dream")
def trigger_dream(full: bool = False, db: Session = Depends(get_db)):
    # منع تشغيل Dream مرتين في نفس الوقت
    running = db.query(DreamLog).filter(DreamLog.status == "running").first()
    if running:
        return {"status": "already_running", "message": "Dream يعمل بالفعل في الخلفية"}

    from dreaming.dream_engine import DreamEngine
    def run():
        DreamEngine().run(full=full)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return {"status": "started", "message": "جاري تشغيل دورة الـ Dreaming في الخلفية"}


@router.get("/dreams")
def list_dreams(db: Session = Depends(get_db)):
    logs = db.query(DreamLog).order_by(desc(DreamLog.started_at)).limit(20).all()
    return [{"id": l.id, "started_at": l.started_at.isoformat(),
             "finished_at": l.finished_at.isoformat() if l.finished_at else None,
             "status": l.status, "conversations": l.conversations_processed,
             "pages_created": l.pages_created, "pages_updated": l.pages_updated,
             "insights": l.insights_found, "summary": l.summary} for l in logs]

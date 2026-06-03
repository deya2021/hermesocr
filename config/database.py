from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database with pgvector extension and run safe migrations."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

    # ── Safe migrations: add new columns if they don't exist ──
    _safe_add_column("conversations", "is_digested", "BOOLEAN NOT NULL DEFAULT FALSE")
    # IngestJob — live-progress fields (added in Phase 2)
    _safe_add_column("ingest_jobs", "processed",     "INTEGER DEFAULT 0")
    _safe_add_column("ingest_jobs", "current_title", "VARCHAR(500)")
    print("✅ Database initialized")


def _safe_add_column(table: str, column: str, definition: str):
    """Add a column to a table if it doesn't already exist."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name=:t AND column_name=:c"
        ), {"t": table, "c": column})
        if not result.fetchone():
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
            conn.commit()
            print(f"✅ Migration: added {table}.{column}")

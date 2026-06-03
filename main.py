import sys
import os
sys.path.insert(0, '/home/opc/webapp')

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from api.routes import router
from config.database import init_db
from dreaming.scheduler import start_scheduler, stop_scheduler
from rich.console import Console

console = Console()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    console.print("\n[bold cyan]🧠 Memory Wiki Starting...[/bold cyan]")
    init_db()
    start_scheduler()
    console.print("[bold green]✅ Ready![/bold green]\n")
    yield
    # Shutdown
    stop_scheduler()
    console.print("\n[yellow]Memory Wiki stopped.[/yellow]")

app = FastAPI(
    title="Memory Wiki API",
    description="Personal AI conversation knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

# API routes
app.include_router(router, prefix="/api")

# Web UI
templates = Jinja2Templates(directory="/home/opc/webapp/web/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/", response_class=HTMLResponse)
async def index_head(request: Request):
    return HTMLResponse(content="", status_code=200)

if __name__ == "__main__":
    import uvicorn
    from config.settings import settings
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        workers=1
    )

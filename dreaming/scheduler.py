from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from config.settings import settings
from rich.console import Console

console = Console()
scheduler = BackgroundScheduler()

def start_scheduler():
    from dreaming.dream_engine import DreamEngine
    
    def nightly_dream():
        console.print("\n⏰ Nightly dreaming triggered by scheduler")
        engine = DreamEngine()
        engine.run()

    scheduler.add_job(
        nightly_dream,
        CronTrigger(hour=settings.dreaming_hour, minute=settings.dreaming_minute),
        id="nightly_dream",
        replace_existing=True
    )
    scheduler.start()
    console.print(f"⏰ Dreaming scheduler started — runs daily at {settings.dreaming_hour:02d}:{settings.dreaming_minute:02d}")

def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()

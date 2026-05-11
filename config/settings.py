from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://memwiki:memwiki123@localhost:5432/memwiki"
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "hermes3:8b"
    
    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "change_this_to_random_string"
    wiki_path: str = "/home/opc/webapp/wiki"
    uploads_path: str = "/home/opc/webapp/uploads"
    logs_path: str = "/home/opc/webapp/logs"
    
    # Dreaming
    dreaming_hour: int = 2
    dreaming_minute: int = 0

    class Config:
        env_file = "/home/opc/webapp/config/.env"

settings = Settings()

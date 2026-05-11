from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from config.models import SourceType

@dataclass
class ParsedMessage:
    role: str           # user / assistant
    content: str
    created_at: Optional[datetime] = None
    order_index: int = 0

@dataclass
class ParsedConversation:
    source: SourceType
    title: str
    original_id: str
    created_at: Optional[datetime]
    messages: List[ParsedMessage]
    raw_file: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseParser(ABC):
    """Base class for all conversation parsers"""

    @abstractmethod
    def can_parse(self, file_path: str, data: dict) -> bool:
        """Check if this parser can handle the given file"""
        pass

    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedConversation]:
        """Parse file and return list of conversations"""
        pass

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return text.strip()

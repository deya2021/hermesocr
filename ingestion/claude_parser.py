import json
import zipfile
from datetime import datetime
from typing import List
from ingestion.base_parser import BaseParser, ParsedConversation, ParsedMessage
from config.models import SourceType

class ClaudeParser(BaseParser):
    """Parser for Claude export (JSON)"""

    def can_parse(self, file_path: str, data: dict = None) -> bool:
        if file_path.endswith('.json') and data:
            if isinstance(data, list) and len(data) > 0:
                return 'chat_messages' in data[0] or 'uuid' in data[0]
        return False

    def parse(self, file_path: str) -> List[ParsedConversation]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._parse_conversations(data, file_path)

    def _parse_conversations(self, data: list, raw_file: str) -> List[ParsedConversation]:
        conversations = []
        items = data if isinstance(data, list) else [data]
        for item in items:
            try:
                conv = self._parse_single(item, raw_file)
                if conv and len(conv.messages) > 0:
                    conversations.append(conv)
            except Exception as e:
                print(f"⚠️  Skipping Claude conversation: {e}")
        return conversations

    def _parse_single(self, item: dict, raw_file: str) -> ParsedConversation:
        title = item.get('name', item.get('title', 'Untitled'))
        original_id = item.get('uuid', item.get('id', ''))

        created_str = item.get('created_at', '')
        try:
            created_at = datetime.fromisoformat(created_str.replace('Z', '+00:00')) if created_str else None
        except:
            created_at = None

        messages = []
        raw_messages = item.get('chat_messages', item.get('messages', []))

        for idx, msg in enumerate(raw_messages):
            role = msg.get('sender', msg.get('role', ''))
            # Claude uses 'human' / 'assistant'
            if role == 'human':
                role = 'user'
            if role not in ('user', 'assistant'):
                continue

            # Content can be string or list
            content = msg.get('text', msg.get('content', ''))
            if isinstance(content, list):
                content = ' '.join([
                    c.get('text', '') if isinstance(c, dict) else str(c)
                    for c in content
                ])
            content = str(content).strip()
            if not content:
                continue

            msg_time_str = msg.get('created_at', '')
            try:
                msg_dt = datetime.fromisoformat(msg_time_str.replace('Z', '+00:00')) if msg_time_str else None
            except:
                msg_dt = None

            messages.append(ParsedMessage(
                role=role,
                content=content,
                created_at=msg_dt,
                order_index=idx
            ))

        return ParsedConversation(
            source=SourceType.CLAUDE,
            title=title,
            original_id=original_id,
            created_at=created_at,
            messages=messages,
            raw_file=raw_file
        )

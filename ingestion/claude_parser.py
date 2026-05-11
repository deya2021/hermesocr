import json
import zipfile
from datetime import datetime
from typing import List
from ingestion.base_parser import BaseParser, ParsedConversation, ParsedMessage
from config.models import SourceType


class ClaudeParser(BaseParser):
    """Parser for Claude export — JSON file or ZIP bundle"""

    # ──────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────

    def can_parse(self, file_path: str, data=None) -> bool:
        # ZIP export from Claude (contains conversations.json)
        if file_path.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path) as z:
                    return 'conversations.json' in z.namelist()
            except Exception:
                return False

        # Raw JSON export
        if file_path.endswith('.json') and data:
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                return 'chat_messages' in first or (
                    'uuid' in first and 'name' in first
                )
        return False

    # ──────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────

    def parse(self, file_path: str) -> List[ParsedConversation]:
        if file_path.endswith('.zip'):
            return self._parse_zip(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._parse_conversations(data, file_path)

    # ──────────────────────────────────────────────────────────
    # ZIP handling
    # ──────────────────────────────────────────────────────────

    def _parse_zip(self, zip_path: str) -> List[ParsedConversation]:
        conversations = []
        with zipfile.ZipFile(zip_path, 'r') as z:
            names = z.namelist()

            # 1. Top-level conversations.json
            if 'conversations.json' in names:
                with z.open('conversations.json') as f:
                    data = json.load(f)
                conversations.extend(self._parse_conversations(data, zip_path))

            # 2. projects/XXXX.json — each project file is a list of convs
            for name in names:
                if name.startswith('projects/') and name.endswith('.json'):
                    try:
                        with z.open(name) as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            conversations.extend(
                                self._parse_conversations(data, zip_path)
                            )
                    except Exception as e:
                        print(f"⚠️  Skipping project file {name}: {e}")

        print(f"   Claude ZIP — found {len(conversations)} conversations total")
        return conversations

    # ──────────────────────────────────────────────────────────
    # Parse conversation list
    # ──────────────────────────────────────────────────────────

    def _parse_conversations(self, data, raw_file: str) -> List[ParsedConversation]:
        conversations = []
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            # Skip project-metadata objects that have no messages
            if 'chat_messages' not in item and 'messages' not in item:
                continue
            try:
                conv = self._parse_single(item, raw_file)
                if conv and len(conv.messages) > 0:
                    conversations.append(conv)
            except Exception as e:
                print(f"⚠️  Skipping Claude conversation: {e}")
        return conversations

    # ──────────────────────────────────────────────────────────
    # Parse a single conversation object
    # ──────────────────────────────────────────────────────────

    def _parse_single(self, item: dict, raw_file: str) -> ParsedConversation:
        title = item.get('name', item.get('title', 'Untitled'))
        original_id = item.get('uuid', item.get('id', ''))

        created_str = item.get('created_at', '')
        try:
            created_at = datetime.fromisoformat(
                created_str.replace('Z', '+00:00')
            ) if created_str else None
        except Exception:
            created_at = None

        messages = []
        raw_messages = item.get('chat_messages', item.get('messages', []))

        for idx, msg in enumerate(raw_messages):
            role = msg.get('sender', msg.get('role', ''))
            if role == 'human':
                role = 'user'
            if role not in ('user', 'assistant'):
                continue

            content = self._extract_text(msg)
            if not content:
                continue

            msg_time_str = msg.get('created_at', '')
            try:
                msg_dt = datetime.fromisoformat(
                    msg_time_str.replace('Z', '+00:00')
                ) if msg_time_str else None
            except Exception:
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

    # ──────────────────────────────────────────────────────────
    # Text extraction — handles all known Claude content formats
    # ──────────────────────────────────────────────────────────

    def _extract_text(self, msg: dict) -> str:
        """
        Claude exports have evolved — try every known layout:

        Format A (old):  msg['text'] = "plain string"
        Format B (new):  msg['text'] = ""
                         msg['content'] = [{"type":"text","text":"..."},...]
        Format C (rare): msg['content'] = "plain string"
        """

        # ── Format A: text field is a non-empty string ──────────────
        text_field = msg.get('text', '')
        if isinstance(text_field, str) and text_field.strip():
            return text_field.strip()

        # ── Format B: content is a list of typed blocks ──────────────
        content_field = msg.get('content', '')
        if isinstance(content_field, list):
            parts = []
            for block in content_field:
                if not isinstance(block, dict):
                    continue
                block_type = block.get('type', '')
                if block_type == 'text':
                    t = block.get('text', '').strip()
                    if t:
                        parts.append(t)
                # tool_use / tool_result — extract input/output text
                elif block_type == 'tool_result':
                    for sub in block.get('content', []):
                        if isinstance(sub, dict) and sub.get('type') == 'text':
                            t = sub.get('text', '').strip()
                            if t:
                                parts.append(t)
            result = '\n\n'.join(parts).strip()
            if result:
                return result

        # ── Format C: content is a plain string ──────────────────────
        if isinstance(content_field, str) and content_field.strip():
            return content_field.strip()

        return ''

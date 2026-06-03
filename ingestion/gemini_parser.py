"""
Gemini / Google AI Studio conversation export parser.

Google AI Studio exports conversations in two known formats:

Format A — Single JSON file (direct export):
    A list of conversation objects, each with:
    {
      "title": "...",
      "createTime": "2024-01-01T00:00:00Z",
      "messages": [
          {"author": "user",  "content": "..."},
          {"author": "model", "content": "..."}
      ]
    }

Format B — ZIP export (Google Takeout style):
    Contains one or more .json files under a conversations/ directory,
    each file is either a single conversation object or a list.

Format C — Google Takeout AI conversations:
    JSON with "conversations" top-level key:
    {
      "conversations": [
          {
            "id": "...",
            "title": "...",
            "messages": [...]
          }
      ]
    }
"""

import json
import zipfile
import os
from datetime import datetime
from typing import List, Optional
from ingestion.base_parser import BaseParser, ParsedConversation, ParsedMessage
from config.models import SourceType


class GeminiParser(BaseParser):
    """Parser for Google AI Studio / Gemini conversation exports."""

    # ──────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────
    def can_parse(self, file_path: str, data=None) -> bool:
        if file_path.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path) as z:
                    names = z.namelist()
                # Google Takeout: usually contains "Gemini Apps Activity.json"
                # or conversations/ directory
                return any(
                    'gemini' in n.lower() or 'google ai' in n.lower()
                    or n.startswith('conversations/')
                    for n in names
                )
            except Exception:
                return False

        if file_path.endswith('.json') and data:
            return self._looks_like_gemini(data)

        return False

    def _looks_like_gemini(self, data) -> bool:
        """Heuristic: detect Gemini JSON structure."""
        # Format C: top-level "conversations" key
        if isinstance(data, dict) and "conversations" in data:
            convs = data["conversations"]
            if isinstance(convs, list) and convs:
                first = convs[0]
                if isinstance(first, dict) and "messages" in first:
                    msgs = first.get("messages", [])
                    if msgs and isinstance(msgs[0], dict):
                        author = msgs[0].get("author", "")
                        return author in ("user", "model", "0", "1")
            return False

        # Format A: list with "createTime" or author="model"
        if isinstance(data, list) and data:
            first = data[0]
            if not isinstance(first, dict):
                return False
            if "createTime" in first and "messages" in first:
                return True
            # messages with author="model"
            msgs = first.get("messages", [])
            if msgs and isinstance(msgs[0], dict):
                author = msgs[0].get("author", "")
                return author in ("model",)
        return False

    # ──────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────
    def parse(self, file_path: str) -> List[ParsedConversation]:
        if file_path.endswith('.zip'):
            return self._parse_zip(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._parse_data(data, file_path)

    # ──────────────────────────────────────────────────────────
    # ZIP handling
    # ──────────────────────────────────────────────────────────
    def _parse_zip(self, zip_path: str) -> List[ParsedConversation]:
        conversations = []
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if not name.endswith('.json'):
                    continue
                try:
                    with z.open(name) as f:
                        data = json.load(f)
                    convs = self._parse_data(data, zip_path)
                    conversations.extend(convs)
                except Exception as e:
                    print(f"⚠️  Gemini ZIP: skipping {name}: {e}")
        print(f"   Gemini ZIP — found {len(conversations)} conversations total")
        return conversations

    # ──────────────────────────────────────────────────────────
    # Parse any supported data structure
    # ──────────────────────────────────────────────────────────
    def _parse_data(self, data, raw_file: str) -> List[ParsedConversation]:
        # Format C: {"conversations": [...]}
        if isinstance(data, dict) and "conversations" in data:
            items = data["conversations"]
        # Format A: [conv1, conv2, ...]
        elif isinstance(data, list):
            items = data
        # Single conversation object
        elif isinstance(data, dict):
            items = [data]
        else:
            return []

        conversations = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                conv = self._parse_single(item, raw_file)
                if conv and len(conv.messages) > 0:
                    conversations.append(conv)
            except Exception as e:
                print(f"⚠️  Skipping Gemini conversation: {e}")
        return conversations

    # ──────────────────────────────────────────────────────────
    # Parse a single conversation object
    # ──────────────────────────────────────────────────────────
    def _parse_single(self, item: dict, raw_file: str) -> Optional[ParsedConversation]:
        title       = item.get("title", item.get("name", "Untitled"))
        original_id = item.get("id", item.get("conversationId", ""))

        # Timestamps
        created_at = self._parse_time(
            item.get("createTime", item.get("created_at", item.get("timestamp", "")))
        )

        # Messages — support multiple field names
        raw_messages = (
            item.get("messages")
            or item.get("turns")
            or item.get("chat_messages")
            or []
        )

        messages: List[ParsedMessage] = []
        for idx, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                continue

            # Role normalisation
            raw_role = msg.get("author", msg.get("role", msg.get("sender", "")))
            role = self._normalise_role(raw_role)
            if role not in ("user", "assistant"):
                continue

            content = self._extract_content(msg)
            if not content:
                continue

            msg_time = self._parse_time(
                msg.get("createTime", msg.get("timestamp", msg.get("created_at", "")))
            )

            messages.append(ParsedMessage(
                role        = role,
                content     = content,
                created_at  = msg_time,
                order_index = idx,
            ))

        if not messages:
            return None

        return ParsedConversation(
            source      = SourceType.GEMINI,
            title       = title,
            original_id = original_id,
            created_at  = created_at,
            messages    = messages,
            raw_file    = raw_file,
        )

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    def _normalise_role(self, raw: str) -> str:
        """Map Gemini role names to user/assistant."""
        raw = str(raw).lower().strip()
        if raw in ("user", "human", "0"):
            return "user"
        if raw in ("model", "assistant", "gemini", "bard", "1"):
            return "assistant"
        return raw

    def _extract_content(self, msg: dict) -> str:
        """Extract text from various Gemini message content formats."""
        # Plain string fields
        for field in ("content", "text", "parts"):
            val = msg.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()
            # parts is sometimes a list of strings or dicts
            if isinstance(val, list):
                parts = []
                for part in val:
                    if isinstance(part, str) and part.strip():
                        parts.append(part.strip())
                    elif isinstance(part, dict):
                        t = part.get("text", "")
                        if t and t.strip():
                            parts.append(t.strip())
                result = "\n\n".join(parts).strip()
                if result:
                    return result

        return ""

    def _parse_time(self, value) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.utcfromtimestamp(value / 1000 if value > 1e10 else value)
            except Exception:
                return None
        if isinstance(value, str):
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    return datetime.strptime(value[:26], fmt)
                except Exception:
                    continue
            try:
                from dateutil import parser as dateparser
                return dateparser.parse(value)
            except Exception:
                return None
        return None

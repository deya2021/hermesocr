"""
ChatGPT conversation export parser.

Supports:
  - conversations.json  (direct export or inside ZIP)
  - ZIP export          (contains conversations.json at root)

Large-file strategy:
  - JSON files > STREAM_THRESHOLD are parsed with ijson (streaming)
    so we never load the full 120MB into RAM at once.
  - Falls back to standard json.load for smaller files.
"""

import json
import zipfile
import os
import tempfile
from datetime import datetime
from typing import List, Iterator
from ingestion.base_parser import BaseParser, ParsedConversation, ParsedMessage
from config.models import SourceType

# Files larger than this are parsed with streaming (ijson)
STREAM_THRESHOLD = 20 * 1024 * 1024   # 20 MB


class ChatGPTParser(BaseParser):
    """Parser for ChatGPT export (conversations.json or ZIP)"""

    def can_parse(self, file_path: str, data: dict = None) -> bool:
        if file_path.endswith('.zip'):
            return True
        if file_path.endswith('.json') and data:
            if isinstance(data, list) and len(data) > 0:
                return 'mapping' in data[0] or 'conversation_id' in data[0]
        return False

    def parse(self, file_path: str) -> List[ParsedConversation]:
        if file_path.endswith('.zip'):
            return self._parse_zip(file_path)
        return list(self._stream_json_file(file_path))

    # ──────────────────────────────────────────────────────────
    # ZIP
    # ──────────────────────────────────────────────────────────
    def _parse_zip(self, zip_path: str) -> List[ParsedConversation]:
        conversations = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name == 'conversations.json':
                    # استخراج إلى ملف مؤقت لدعم streaming
                    with tempfile.NamedTemporaryFile(
                        suffix='.json', delete=False, dir='/tmp'
                    ) as tmp:
                        tmp_path = tmp.name
                        with zf.open(name) as src:
                            while True:
                                chunk = src.read(4 * 1024 * 1024)  # 4MB chunks
                                if not chunk:
                                    break
                                tmp.write(chunk)
                    try:
                        conversations = list(self._stream_json_file(tmp_path, raw_file=zip_path))
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                    break
        return conversations

    # ──────────────────────────────────────────────────────────
    # Streaming JSON parser (handles 120MB+ files)
    # ──────────────────────────────────────────────────────────
    def _stream_json_file(
        self, json_path: str, raw_file: str = None
    ) -> Iterator[ParsedConversation]:
        """
        Parse conversations.json in a memory-efficient way.
        - Small files  (< 20MB): standard json.load
        - Large files (>= 20MB): ijson streaming parser
        """
        raw_file = raw_file or json_path
        file_size = os.path.getsize(json_path)

        if file_size < STREAM_THRESHOLD:
            # ── Standard path ────────────────────────────────
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                try:
                    conv = self._parse_single(item, raw_file)
                    if conv and conv.messages:
                        yield conv
                except Exception as e:
                    print(f"⚠️  Skipping conversation: {e}")
        else:
            # ── Streaming path (ijson) ────────────────────────
            yield from self._stream_with_ijson(json_path, raw_file)

    def _stream_with_ijson(
        self, json_path: str, raw_file: str
    ) -> Iterator[ParsedConversation]:
        """
        Use ijson to iterate over top-level array items one by one.
        Memory usage stays constant regardless of file size.
        """
        try:
            import ijson
        except ImportError:
            # ijson غير مثبّت — نستخدم fallback بسيط
            print("⚠️  ijson not installed, using chunked fallback")
            yield from self._stream_chunked_fallback(json_path, raw_file)
            return

        print(f"   📡 Streaming {json_path} ({os.path.getsize(json_path)//1024//1024}MB) with ijson")
        with open(json_path, 'rb') as f:
            parser = ijson.items(f, 'item')
            for item in parser:
                try:
                    conv = self._parse_single(item, raw_file)
                    if conv and conv.messages:
                        yield conv
                except Exception as e:
                    print(f"⚠️  Skipping conversation: {e}")

    def _stream_chunked_fallback(
        self, json_path: str, raw_file: str
    ) -> Iterator[ParsedConversation]:
        """
        Fallback when ijson is unavailable:
        Read the file in text chunks, splitting on top-level array items.
        Works for well-formed ChatGPT exports.
        """
        print("   📡 Using chunked JSON fallback parser")
        with open(json_path, 'r', encoding='utf-8') as f:
            # تخطي '[' الافتتاحية
            content = f.read()

        # فصل العناصر بشكل آمن
        depth    = 0
        start    = None
        in_str   = False
        escape   = False

        for i, ch in enumerate(content):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        item = json.loads(content[start:i+1])
                        conv = self._parse_single(item, raw_file)
                        if conv and conv.messages:
                            yield conv
                    except Exception as e:
                        print(f"⚠️  Skipping: {e}")
                    start = None

    # ──────────────────────────────────────────────────────────
    # Parse a single conversation object
    # ──────────────────────────────────────────────────────────
    def _parse_single(self, item: dict, raw_file: str) -> ParsedConversation:
        title       = item.get('title', 'Untitled')
        original_id = item.get('id', item.get('conversation_id', ''))

        create_time = item.get('create_time')
        created_at  = datetime.fromtimestamp(create_time) if create_time else None

        mapping     = item.get('mapping', {})
        ordered_ids = self._get_ordered_ids(mapping)

        messages = []
        for idx, node_id in enumerate(ordered_ids):
            node = mapping.get(node_id, {})
            msg  = node.get('message')
            if not msg:
                continue

            role = msg.get('author', {}).get('role', '')
            if role not in ('user', 'assistant'):
                continue

            content_parts = msg.get('content', {}).get('parts', [])
            content = ' '.join(
                p if isinstance(p, str) else
                (p.get('text', '') if isinstance(p, dict) else '')
                for p in content_parts
            ).strip()

            if not content:
                continue

            msg_time = msg.get('create_time')
            msg_dt   = datetime.fromtimestamp(msg_time) if msg_time else None

            messages.append(ParsedMessage(
                role        = role,
                content     = content,
                created_at  = msg_dt,
                order_index = idx,
            ))

        return ParsedConversation(
            source      = SourceType.CHATGPT,
            title       = title,
            original_id = original_id,
            created_at  = created_at,
            messages    = messages,
            raw_file    = raw_file,
        )

    def _get_ordered_ids(self, mapping: dict) -> list:
        """Traverse the linked list to get ordered message IDs."""
        all_ids = set(mapping.keys())
        root    = None
        for node_id, node in mapping.items():
            parent = node.get('parent')
            if not parent or parent not in all_ids:
                root = node_id
                break

        if not root:
            return list(mapping.keys())

        ordered = []
        stack   = [root]
        while stack:
            current  = stack.pop()
            ordered.append(current)
            children = mapping.get(current, {}).get('children', [])
            for child in reversed(children):
                stack.append(child)
        return ordered

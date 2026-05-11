import json
import zipfile
import os
from datetime import datetime
from typing import List
from dateutil import parser as dateparser
from ingestion.base_parser import BaseParser, ParsedConversation, ParsedMessage
from config.models import SourceType

class ChatGPTParser(BaseParser):
    """Parser for ChatGPT export (conversations.json or ZIP)"""

    def can_parse(self, file_path: str, data: dict = None) -> bool:
        if file_path.endswith('.zip'):
            return True
        if file_path.endswith('.json') and data:
            # ChatGPT format: list of conversations with 'mapping' key
            if isinstance(data, list) and len(data) > 0:
                return 'mapping' in data[0] or 'conversation_id' in data[0]
        return False

    def parse(self, file_path: str) -> List[ParsedConversation]:
        conversations = []

        if file_path.endswith('.zip'):
            conversations = self._parse_zip(file_path)
        elif file_path.endswith('.json'):
            conversations = self._parse_json_file(file_path)

        return conversations

    def _parse_zip(self, zip_path: str) -> List[ParsedConversation]:
        conversations = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name == 'conversations.json':
                    with zf.open(name) as f:
                        data = json.load(f)
                        conversations = self._parse_conversations(data, zip_path)
        return conversations

    def _parse_json_file(self, json_path: str) -> List[ParsedConversation]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._parse_conversations(data, json_path)

    def _parse_conversations(self, data: list, raw_file: str) -> List[ParsedConversation]:
        conversations = []
        for item in data:
            try:
                conv = self._parse_single(item, raw_file)
                if conv and len(conv.messages) > 0:
                    conversations.append(conv)
            except Exception as e:
                print(f"⚠️  Skipping conversation: {e}")
        return conversations

    def _parse_single(self, item: dict, raw_file: str) -> ParsedConversation:
        title = item.get('title', 'Untitled')
        original_id = item.get('id', item.get('conversation_id', ''))
        
        # Parse creation time
        create_time = item.get('create_time')
        created_at = datetime.fromtimestamp(create_time) if create_time else None

        # Extract messages from mapping
        messages = []
        mapping = item.get('mapping', {})
        
        # Build ordered list from linked list structure
        ordered_ids = self._get_ordered_ids(mapping)
        
        for idx, node_id in enumerate(ordered_ids):
            node = mapping.get(node_id, {})
            msg = node.get('message')
            if not msg:
                continue
            
            role = msg.get('author', {}).get('role', '')
            if role not in ('user', 'assistant'):
                continue

            # Extract text content
            content_parts = msg.get('content', {}).get('parts', [])
            content = ' '.join([
                p if isinstance(p, str) else ''
                for p in content_parts
            ]).strip()

            if not content:
                continue

            msg_time = msg.get('create_time')
            msg_dt = datetime.fromtimestamp(msg_time) if msg_time else None

            messages.append(ParsedMessage(
                role=role,
                content=content,
                created_at=msg_dt,
                order_index=idx
            ))

        return ParsedConversation(
            source=SourceType.CHATGPT,
            title=title,
            original_id=original_id,
            created_at=created_at,
            messages=messages,
            raw_file=raw_file
        )

    def _get_ordered_ids(self, mapping: dict) -> list:
        """Traverse the linked list to get ordered message IDs"""
        # Find root node (no parent or parent not in mapping)
        children_map = {}
        all_ids = set(mapping.keys())
        
        root = None
        for node_id, node in mapping.items():
            parent = node.get('parent')
            if not parent or parent not in all_ids:
                root = node_id
                break
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(node_id)

        if not root:
            return list(mapping.keys())

        # DFS traversal
        ordered = []
        stack = [root]
        while stack:
            current = stack.pop()
            ordered.append(current)
            children = mapping.get(current, {}).get('children', [])
            # Push in reverse to maintain order
            for child in reversed(children):
                stack.append(child)

        return ordered

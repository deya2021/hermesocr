import ollama
from typing import List, Dict
from config.settings import settings


class LLMAgent:
    """
    LLM agent for wiki generation.
    Uses llama3.2:1b (fast on CPU) with short prompts.
    hermes3:8b is kept as fallback for future GPU use.
    """

    def __init__(self):
        # llama3.2:1b is ~5x faster than hermes3:8b on CPU
        self.model  = "llama3.2:1b"
        self.host   = settings.ollama_host
        self.client = ollama.Client(host=self.host)

        # Max tokens per call — keep short to stay fast on CPU
        self.fast_opts   = {"temperature": 0.1, "num_predict": 300}
        self.normal_opts = {"temperature": 0.3, "num_predict": 500}
        self.creative_opts = {"temperature": 0.5, "num_predict": 600}

    # ──────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────

    def _chat(self, system: str, user: str, opts: dict = None) -> str:
        if opts is None:
            opts = self.normal_opts
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options=opts,
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            return f"<!-- LLM ERROR: {e} -->"

    # ──────────────────────────────────────────────────────────
    # 1. Extract topics  (fast — JSON array)
    # ──────────────────────────────────────────────────────────

    def extract_topics(self, text: str) -> List[str]:
        system = (
            "Extract the main topics from the conversation text. "
            "Reply with ONLY a JSON array of short strings (2-4 words each), max 5. "
            'Example: ["python async","database design","API security"]'
        )
        result = self._chat(system, text[:1200], self.fast_opts)
        try:
            import json
            s = result.find("[")
            e = result.rfind("]") + 1
            if s >= 0 and e > s:
                return json.loads(result[s:e])
        except Exception:
            pass
        return []

    # ──────────────────────────────────────────────────────────
    # 2. Summarize conversation → source wiki page
    # ──────────────────────────────────────────────────────────

    def summarize_conversation(self, title: str, messages_text: str) -> str:
        system = (
            "You are a wiki writer. Write a short wiki page for this AI conversation.\n"
            "Use Markdown. Include:\n"
            "## Summary\n## Key Points\n## Topics\n"
            "Be concise. Max 400 words."
        )
        user = f"Title: {title}\n\n{messages_text[:2000]}"
        return self._chat(system, user, self.normal_opts)

    # ──────────────────────────────────────────────────────────
    # 3. Generate topic page
    # ──────────────────────────────────────────────────────────

    def generate_topic_page(self, topic: str, chunks: List[str]) -> str:
        system = (
            "You are a personal knowledge wiki writer.\n"
            "Write a wiki page for the given topic using the conversation excerpts.\n"
            "Markdown format:\n"
            "## Overview\n## Key Concepts\n## Practical Notes\n## Open Questions\n"
            "Max 500 words. Cite sources as (conv_N)."
        )
        excerpts = "\n---\n".join(chunks[:5])
        user = f"Topic: {topic}\n\nExcerpts:\n{excerpts[:2500]}"
        return self._chat(system, user, self.normal_opts)

    # ──────────────────────────────────────────────────────────
    # 4. Find cross-topic connections  (dreaming phase)
    # ──────────────────────────────────────────────────────────

    def find_connections(self, topics: List[str], summaries: List[str]) -> str:
        system = (
            "You are analyzing a personal knowledge base to find hidden connections.\n"
            "Markdown format:\n"
            "## Unexpected Connections\n## Recurring Patterns\n## Emergent Insights\n"
            "Be insightful and concise. Max 400 words."
        )
        topic_str   = ", ".join(topics[:20])
        summary_str = "\n\n".join(s[:300] for s in summaries[:8])
        user = f"Topics: {topic_str}\n\nSummaries:\n{summary_str}"
        return self._chat(system, user, self.creative_opts)

    # ──────────────────────────────────────────────────────────
    # 5. Compress / merge knowledge into existing page
    # ──────────────────────────────────────────────────────────

    def compress_knowledge(self, old_page: str, new_chunks: List[str]) -> str:
        system = (
            "Merge new information into this wiki page.\n"
            "Keep existing content, add new insights, remove duplicates.\n"
            "Mark additions with '🆕'. Return full updated Markdown page. Max 600 words."
        )
        new_text = "\n---\n".join(new_chunks[:3])
        user = f"EXISTING PAGE:\n{old_page[:1500]}\n\nNEW INFO:\n{new_text[:1000]}"
        return self._chat(system, user, self.normal_opts)

    # ──────────────────────────────────────────────────────────
    # 6. Generate master overview
    # ──────────────────────────────────────────────────────────

    def generate_overview(self, all_topics: List[str], stats: Dict) -> str:
        system = (
            "Write a master overview page for a personal AI conversation wiki.\n"
            "Markdown format:\n"
            "## My Knowledge Universe\n## Most Explored Topics\n"
            "## Knowledge Clusters\n## Learning Journey\n"
            "Max 500 words."
        )
        user = (
            f"Stats: {stats.get('conversations',0)} conversations, "
            f"{stats.get('topics',0)} topics, "
            f"date range: {stats.get('date_range','unknown')}\n\n"
            f"Topics: {', '.join(all_topics[:40])}"
        )
        return self._chat(system, user, self.creative_opts)

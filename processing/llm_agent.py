import ollama
from typing import List, Dict, Optional
from config.settings import settings

class LLMAgent:
    """Hermes3 agent for wiki generation and analysis"""

    def __init__(self):
        self.model = settings.ollama_llm_model
        self.host = settings.ollama_host
        self.client = ollama.Client(host=self.host)

    def _chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                options={"temperature": temperature}
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"<!-- LLM ERROR: {e} -->"

    def extract_topics(self, conversation_text: str) -> List[str]:
        """Extract main topics from a conversation"""
        system = """You are a knowledge extraction assistant.
Extract the main topics discussed in this conversation.
Return ONLY a JSON array of short topic strings (2-4 words each), max 5 topics.
Example: ["python decorators", "async programming", "database optimization"]"""

        result = self._chat(system, conversation_text[:2000])
        try:
            import json
            # Extract JSON array from response
            start = result.find('[')
            end = result.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except:
            pass
        return []

    def summarize_conversation(self, title: str, messages_text: str) -> str:
        """Generate a wiki page for a single conversation"""
        system = """You are a personal knowledge wiki writer.
Write a concise wiki page summarizing this AI conversation.
Format as Markdown with:
- ## Summary (2-3 sentences)
- ## Key Points (bullet list)
- ## Topics
- ## Notable Quotes (best 1-2 exchanges)

Be factual. Only describe what is explicitly in the conversation.
Mark uncertain claims with > TODO-VERIFY"""

        user = f"Conversation title: {title}\n\n{messages_text[:3000]}"
        return self._chat(system, user)

    def generate_topic_page(self, topic: str, chunks: List[str]) -> str:
        """Generate a wiki page for a topic aggregated from many conversations"""
        system = """You are a personal knowledge wiki writer building a Karpathy-style LLM wiki.
Generate a comprehensive wiki page for this topic based on the provided conversation excerpts.
Format as Markdown:
- ## Overview
- ## Key Concepts
- ## Practical Applications
- ## Open Questions (things still unclear)
- ## Sources (list conversation references)

Cite sources as (conv_N). Only claim what's in the excerpts.
Mark gaps with > TODO-VERIFY"""

        chunks_text = "\n\n---\n\n".join(chunks[:10])
        user = f"Topic: {topic}\n\nRelevant conversation excerpts:\n\n{chunks_text}"
        return self._chat(system, user, temperature=0.4)

    def find_connections(self, topics: List[str], summaries: List[str]) -> str:
        """Dreaming: find non-obvious connections between topics"""
        system = """You are a deep knowledge synthesizer performing memory consolidation (like dreaming).
Analyze these topics and conversation summaries.
Find NON-OBVIOUS connections, patterns, and emergent insights that span multiple conversations.
Format as Markdown:
- ## Unexpected Connections
- ## Recurring Patterns
- ## Emergent Insights
- ## Knowledge Gaps to Explore

Think deeply. Surface insights that wouldn't be visible from any single conversation."""

        topics_text = ", ".join(topics)
        summaries_text = "\n\n".join(summaries[:15])
        user = f"Topics found: {topics_text}\n\nConversation summaries:\n{summaries_text}"
        return self._chat(system, user, temperature=0.7)

    def compress_knowledge(self, old_page: str, new_chunks: List[str]) -> str:
        """Dreaming: merge new knowledge into existing wiki page"""
        system = """You are updating a personal knowledge wiki page with new information.
Merge the new conversation excerpts into the existing wiki page.
- Keep all valuable existing content
- Add new insights and information
- Remove redundant or outdated content
- Highlight what changed with > 🆕 NEW prefix
Return the complete updated wiki page in Markdown."""

        new_text = "\n\n---\n\n".join(new_chunks[:5])
        user = f"EXISTING PAGE:\n{old_page}\n\nNEW EXCERPTS TO MERGE:\n{new_text}"
        return self._chat(system, user, temperature=0.3)

    def generate_overview(self, all_topics: List[str], stats: Dict) -> str:
        """Generate the master overview page"""
        system = """You are writing the master overview page for a personal AI knowledge wiki.
This wiki contains all conversations the user has had with AI assistants.
Write an insightful overview page in Markdown:
- ## My Knowledge Universe
- ## Most Explored Topics
- ## Knowledge Clusters (groups of related topics)  
- ## Learning Journey (timeline narrative)
- ## Recommended Deep Dives"""

        user = f"""Stats:
- Total conversations: {stats.get('conversations', 0)}
- Total topics: {stats.get('topics', 0)}
- Date range: {stats.get('date_range', 'unknown')}

All topics: {', '.join(all_topics[:50])}"""
        return self._chat(system, user, temperature=0.5)

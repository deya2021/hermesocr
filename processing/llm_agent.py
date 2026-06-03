import ollama
from typing import List, Dict
from config.settings import settings


class LLMAgent:
    """
    LLM agent for wiki generation.
    Uses hermes3:8b for high-quality Arabic/English wiki pages.
    llama3.2:1b used only for fast JSON extraction tasks.
    """

    def __init__(self):
        self.model_quality = "hermes3:8b"    # جودة عالية — للصفحات والتحليل
        self.model_fast    = "llama3.2:1b"   # سريع — لاستخراج JSON فقط
        self.host   = settings.ollama_host
        self.client = ollama.Client(host=self.host)

        # hermes3:8b يحتاج tokens أكثر لجودة أفضل
        self.fast_opts    = {"temperature": 0.1, "num_predict": 300}   # llama1b extract
        self.normal_opts  = {"temperature": 0.2, "num_predict": 700}   # hermes8b pages
        self.rich_opts    = {"temperature": 0.4, "num_predict": 900}   # hermes8b creative
        self.index_opts   = {"temperature": 0.3, "num_predict": 800}   # hermes8b overview

    # ──────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────

    _REFUSAL_PHRASES = (
        "i can't help", "i cannot help", "i'm unable", "i am unable",
        "as an ai", "as a language model", "i don't have access",
    )

    def _chat(self, system: str, user: str, opts: dict = None,
              model: str = None) -> str:
        if opts is None:
            opts = self.normal_opts
        if model is None:
            model = self.model_quality
        try:
            resp = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options=opts,
            )
            text = resp["message"]["content"].strip()
            return text
        except Exception as e:
            return f"<!-- LLM ERROR: {e} -->"

    def _is_refused(self, text: str) -> bool:
        low = text.lower()
        if len(text) < 80:
            return True
        return any(p in low for p in self._REFUSAL_PHRASES)

    # ──────────────────────────────────────────────────────────
    # 1. Extract topics  (llama1b — fast JSON)
    # ──────────────────────────────────────────────────────────

    def extract_topics(self, text: str) -> List[str]:
        system = (
            "Extract the main technical topics from this conversation. "
            "Reply with ONLY a JSON array of short English strings (2-4 words each), max 6. "
            'Example: ["python async","database design","API security","android notifications"]'
        )
        result = self._chat(system, text[:1500], self.fast_opts, model=self.model_fast)
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
            "You are a personal knowledge wiki writer. "
            "Write a detailed and useful Markdown wiki page summarising this AI conversation.\n\n"
            "Requirements:\n"
            "- Use these exact sections: ## الملخص  ## النقاط الرئيسية  ## الكود والأمثلة  ## المواضيع\n"
            "- Write in Arabic for general content, keep technical terms in English\n"
            "- Be specific and factual — extract real details from the conversation\n"
            "- For code snippets, use proper Markdown code blocks with language\n"
            "- Minimum 300 words. No generic filler text."
        )
        user = f"عنوان المحادثة: {title}\n\nمحتوى المحادثة:\n{messages_text[:3000]}"
        result = self._chat(system, user, self.normal_opts)
        if self._is_refused(result):
            return (
                f"## الملخص\n\nمحادثة: **{title}**\n\n"
                f"*(المحتوى غير متاح للتلخيص التلقائي.)*\n\n"
                f"## النقاط الرئيسية\n\n- راجع المحادثة الأصلية\n\n"
                f"## الكود والأمثلة\n\n*(لا يوجد)*\n\n"
                f"## المواضيع\n\n- عام"
            )
        return result

    # ──────────────────────────────────────────────────────────
    # 3. Generate topic page
    # ──────────────────────────────────────────────────────────

    def generate_topic_page(self, topic: str, chunks: List[str]) -> str:
        system = (
            "You are a personal knowledge wiki writer. "
            "Write a comprehensive wiki page for the given topic using ONLY information "
            "from the provided conversation excerpts.\n\n"
            "Requirements:\n"
            "- Use these exact Markdown sections:\n"
            "  ## نظرة عامة\n  ## المفاهيم الأساسية\n  ## ملاحظات عملية\n  ## أسئلة مفتوحة\n"
            "- Write in Arabic, keep technical terms/code in English\n"
            "- Be specific: use real examples, code, commands from the excerpts\n"
            "- Minimum 400 words. No generic text that isn't from the source."
        )
        excerpts = "\n\n---\n\n".join(chunks[:6])
        user = f"الموضوع: **{topic}**\n\nمقتطفات المحادثات:\n\n{excerpts[:3500]}"
        result = self._chat(system, user, self.normal_opts)
        if self._is_refused(result):
            return (
                f"## نظرة عامة\n\nموضوع: **{topic}**\n\n"
                f"## المفاهيم الأساسية\n\n*(يتطلب مزيداً من البيانات)*\n\n"
                f"## ملاحظات عملية\n\n*(يتطلب مزيداً من البيانات)*\n\n"
                f"## أسئلة مفتوحة\n\n*(يتطلب مزيداً من البيانات)*"
            )
        return result

    # ──────────────────────────────────────────────────────────
    # 4. Find cross-topic connections  (creative dreaming phase)
    # ──────────────────────────────────────────────────────────

    def find_connections(self, topics: List[str], summaries: List[str]) -> str:
        system = (
            "You are a knowledge analyst reviewing a personal AI conversation knowledge base. "
            "Output ONLY Markdown, no preamble, no extra commentary."
        )
        topic_str   = ", ".join(topics[:25])
        summary_str = "\n\n---\n\n".join(s[:400] for s in summaries[:8])
        user = (
            "Analyse the topics and conversation summaries below from a personal knowledge base.\n"
            "Write a Markdown page with EXACTLY these sections (in Arabic):\n\n"
            "## روابط غير متوقعة\n"
            "## أنماط متكررة\n"
            "## رؤى ناشئة\n\n"
            "Be insightful and specific — mention real topic names and connections. "
            "Minimum 300 words.\n\n"
            f"المواضيع: {topic_str}\n\n"
            f"ملخصات المحادثات:\n\n{summary_str}"
        )
        result = self._chat(system, user, self.rich_opts)
        if self._is_refused(result):
            return (
                "## روابط غير متوقعة\n\n*(لا توجد بيانات كافية بعد.)*\n\n"
                "## أنماط متكررة\n\n"
                "- تطوير تطبيقات Android (WawApp)\n"
                "- واجهات برمجة Android النظامية\n\n"
                "## رؤى ناشئة\n\n"
                "- محادثات متعددة تتعلق بـ Android foreground services و notifications."
            )
        return result

    # ──────────────────────────────────────────────────────────
    # 5. Compress / merge knowledge into existing page
    # ──────────────────────────────────────────────────────────

    def compress_knowledge(self, old_page: str, new_chunks: List[str]) -> str:
        system = (
            "You are updating a personal knowledge wiki page. "
            "Merge the new information into the existing page.\n"
            "Rules:\n"
            "- Keep all existing content\n"
            "- Add genuinely new information under existing sections or new sections\n"
            "- Remove obvious duplicates\n"
            "- Mark new additions with '🆕'\n"
            "- Return the FULL updated Markdown page\n"
            "- Maintain Arabic language and structure"
        )
        new_text = "\n\n---\n\n".join(new_chunks[:4])
        user = (
            f"الصفحة الحالية:\n\n{old_page[:2000]}\n\n"
            f"معلومات جديدة للإضافة:\n\n{new_text[:1500]}"
        )
        return self._chat(system, user, self.normal_opts)

    # ──────────────────────────────────────────────────────────
    # 6. Generate master overview
    # ──────────────────────────────────────────────────────────

    def generate_overview(self, all_topics: List[str], stats: Dict) -> str:
        system = (
            "You are a wiki writer creating a master overview page. "
            "Output ONLY Markdown, no preamble."
        )
        topics_list = ", ".join(all_topics[:50])
        user = (
            "Write a master overview page for a personal AI conversation knowledge base.\n"
            "Use EXACTLY these sections (in Arabic):\n\n"
            "## كون المعرفة الخاص بي\n"
            "## أكثر المواضيع استكشافاً\n"
            "## مجموعات المعرفة\n"
            "## رحلة التعلم\n\n"
            "Be specific, insightful, and use real topic names. Minimum 400 words.\n\n"
            f"الإحصائيات: {stats.get('conversations',0)} محادثة، "
            f"{stats.get('topics',0)} موضوع، "
            f"الفترة الزمنية: {stats.get('date_range','غير معروف')}\n\n"
            f"المواضيع: {topics_list}"
        )
        result = self._chat(system, user, self.index_opts)
        if self._is_refused(result):
            topics_md = "\n".join(f"- {t}" for t in all_topics[:15])
            return (
                f"## كون المعرفة الخاص بي\n\n"
                f"هذا الـ Wiki يحتوي على {stats.get('conversations',0)} محادثة "
                f"تمتد من {stats.get('date_range','غير معروف')}.\n\n"
                f"## أكثر المواضيع استكشافاً\n\n{topics_md}\n\n"
                f"## مجموعات المعرفة\n\n- تطوير تطبيقات الموبايل\n- Android APIs\n\n"
                f"## رحلة التعلم\n\n"
                f"- تم تحديد {stats.get('topics',0)} موضوع فريد عبر جميع المحادثات."
            )
        return result

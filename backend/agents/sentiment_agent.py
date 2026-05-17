from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json
from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)
from backend.utils.retry import async_retry


def extract_json_text(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start == -1:
            continue

        depth = 0
        for idx, ch in enumerate(text[start:], start=start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
    return None


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    json_text = extract_json_text(text)
    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None

class SentimentPriorityAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=0.1
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sentiment and priority analyzer for customer support.
            Analyze the customer query and determine:
            
            SENTIMENT: positive, neutral, negative, very_negative
            URGENCY: low, medium, high, critical
            EMOTIONS: angry, frustrated, confused, anxious, calm, satisfied
            PRIORITY_SCORE: 1-10 (1=lowest, 10=highest)
            
            Return ONLY valid JSON object with no surrounding text or markdown.
            Example:
            {{
                "sentiment": "negative",
                "urgency": "high",
                "emotions": ["angry", "frustrated"],
                "priority_score": 8,
                "escalation_needed": true,
                "reason": "Customer is angry about delayed shipping"
            }}"""),
            ("human", "Query: {query}\nIntent: {intent}")
        ])
    
    async def analyze(self, query: str, intent: str) -> Dict[str, Any]:
        logger.debug(f"SentimentPriorityAgent.analyze called: query={query[:120]} intent={intent}")
        chain = self.prompt | self.llm

        response = await async_retry(
            lambda: chain.ainvoke({
                "query": query,
                "intent": intent
            }),
            retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )
        
        text = response.content
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")

        result = parse_json_response(text)
        if result is None:
            logger.error(
                "Sentiment analysis parse failed, using fallback. Raw response: %s",
                text.replace("\n", " ")[:1000]
            )
            result = {
                "sentiment": "neutral",
                "urgency": "medium",
                "emotions": [],
                "priority_score": 5,
                "escalation_needed": False,
                "reason": "Unable to determine sentiment"
            }

        return result
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from backend.config import config
import json
from backend.utils.logger import get_logger
from backend.utils.retry import async_retry

logger = get_logger(__name__)

class ResponseGenerationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=0.7
        )
    
    async def generate_response(
        self,
        query: str,
        intent: str,
        sentiment_data: Dict[str, Any],
        rag_data: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        
        # Adjust response style based on sentiment
        style_instructions = {
            "very_negative": "Be extremely empathetic, apologize sincerely, offer immediate solutions",
            "negative": "Be understanding and helpful, acknowledge their frustration",
            "neutral": "Be professional and informative",
            "positive": "Be friendly and appreciative"
        }
        
        sentiment = sentiment_data.get("sentiment", "neutral")
        style = style_instructions.get(sentiment, style_instructions["neutral"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional customer support agent.
            
            Response Style: {style}
            Language: {language}
            
            Generate a response that:
            1. Addresses the customer's specific issue ({intent})
            2. Uses information from the knowledge base
            3. Shows appropriate empathy based on sentiment
            4. Provides clear next steps
            5. Is professional but friendly
            
            Return JSON format:
            {{
                "response_text": "main response here",
                "response_style": "{style}",
                "suggested_actions": ["action1", "action2"],
                "requires_escalation": false,
                "confidence": 0.95
            }}"""),
            ("human", """Customer Query: {query}
            Intent: {intent}
            Sentiment: {sentiment}
            Priority Score: {priority}
            Knowledge Base Context: {context}
            
            Generate response:""")
        ])
        
        chain = prompt | self.llm
        logger.debug("ResponseGenerationAgent.generate_response invoking LLM")

        response = await async_retry(
            lambda: chain.ainvoke({
                "style": style,
                "language": language,
                "intent": intent,
                "query": query,
                "sentiment": sentiment,
                "priority": sentiment_data.get("priority_score", 5),
                "context": rag_data.get("answer", "No context available")
            }),
            retries=4,
            initial_delay=1.0,
            backoff_factor=2.0,
        )
        
        try:
            result = json.loads(response.content)
        except Exception:
            logger.exception("Failed to parse LLM response for generate_response, using fallback")
            result = {
                "response_text": response.content,
                "response_style": style,
                "suggested_actions": [],
                "requires_escalation": False,
                "confidence": 0.5
            }
        
        return result
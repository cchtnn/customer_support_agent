from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json
from backend.config import config
from backend.utils.logger import get_logger
from backend.utils.retry import async_retry

logger = get_logger(__name__)

class IntentClassificationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=0.1
        )
        
        self.intents = [
            "refund_request", "shipping_delay", "password_reset", 
            "order_tracking", "invoice_request", "complaint", 
            "product_inquiry", "account_issue", "cancel_order",
            "return_policy", "human_escalation"
        ]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classification expert for customer support.
            Classify the customer query into one of these intents: {intents}
            
            Also determine the category:
            - BILLING: refund, invoice, payment
            - ORDER: tracking, shipping, cancellation, return
            - ACCOUNT: password, login, profile
            - TECHNICAL: product issues, bugs
            - GENERAL: complaints, inquiries
            
            Return JSON format:
            {{
                "intent": "selected_intent",
                "category": "selected_category",
                "confidence": 0.95,
                "keywords": ["key", "words", "found"]
            }}"""),
            ("human", "{query}")
        ])
    
    async def classify(self, query: str) -> Dict[str, Any]:
        logger.debug(f"IntentClassificationAgent.classify called: query={query[:120]}")
        chain = self.prompt | self.llm

        response = await async_retry(
            lambda: chain.ainvoke({
                "intents": ", ".join(self.intents),
                "query": query
            }),
            retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )
        
        try:
            result = json.loads(response.content)
        except Exception:
            logger.exception("Intent classification failed, using fallback")
            result = {
                "intent": "general_inquiry",
                "category": "GENERAL",
                "confidence": 0.5,
                "keywords": []
            }
        
        return result
    
    # intent_classifier.py

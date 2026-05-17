from typing import Dict, Any
from datetime import datetime
import uuid
import json
from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class EscalationAgent:
    def __init__(self):
        self.tickets = []
        
    async def should_escalate(
        self,
        query: str,
        intent: str,
        sentiment_data: Dict[str, Any],
        response_data: Dict[str, Any],
        conversation_history: list = None
    ) -> Dict[str, Any]:
        
        escalation_reasons = []
        priority_score = sentiment_data.get("priority_score", 0)
        sentiment = sentiment_data.get("sentiment", "neutral")
        response_confidence = response_data.get("confidence", 1.0)
        
        # Check escalation conditions
        if priority_score >= config.ESCALATION_PRIORITY_THRESHOLD:
            escalation_reasons.append(f"High priority score: {priority_score}/10")
        
        if sentiment in ["very_negative", "angry"]:
            escalation_reasons.append(f"Critical sentiment detected: {sentiment}")
        
        if response_confidence < config.CONFIDENCE_THRESHOLD:
            escalation_reasons.append(f"Low response confidence: {response_confidence}")
        
        if intent == "human_escalation":
            escalation_reasons.append("Customer explicitly requested human agent")
        
        # Check for repeated issues
        if conversation_history and len(conversation_history) >= 3:
            escalation_reasons.append("Multiple interactions on same issue")
        
        should_escalate = len(escalation_reasons) > 0

        logger.debug(f"Escalation check: priority_score={priority_score} sentiment={sentiment} response_confidence={response_confidence} reasons={escalation_reasons}")
        
        if should_escalate:
            ticket = self.create_support_ticket(query, intent, sentiment_data, escalation_reasons)
            logger.info(f"Created escalation ticket {ticket['ticket_id']} for query")
            return {
                "should_escalate": True,
                "reasons": escalation_reasons,
                "ticket": ticket,
                "priority": "high" if priority_score > 7 else "medium",
                "assigned_team": "customer_support"
            }
        
        return {
            "should_escalate": False,
            "reasons": [],
            "ticket": None
        }
    
    def create_support_ticket(self, query: str, intent: str, sentiment_data: Dict, reasons: list) -> Dict:
        ticket = {
            "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "created_at": datetime.now().isoformat(),
            "customer_query": query,
            "intent": intent,
            "priority": sentiment_data.get("priority_score", 5),
            "sentiment": sentiment_data.get("sentiment", "neutral"),
            "escalation_reasons": reasons,
            "status": "open",
            "assigned_to": None,
            "resolution_deadline": None
        }
        
        self.tickets.append(ticket)
        return ticket
from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import pandas as pd
import json

class AnalyticsAgent:
    def __init__(self):
        self.conversations = []
        self.tickets = []
        
    def track_conversation(self, conversation_data: Dict[str, Any]):
        conversation_data["timestamp"] = datetime.now().isoformat()
        self.conversations.append(conversation_data)
        
    def generate_analytics(self, time_range_days: int = 7) -> Dict[str, Any]:
        cutoff_date = datetime.now() - timedelta(days=time_range_days)
        
        filtered_conversations = [
            c for c in self.conversations 
            if datetime.fromisoformat(c["timestamp"]) > cutoff_date
        ]
        
        if not filtered_conversations:
            return {"message": "No data available for the selected time range"}
        
        # Intent distribution
        intents = [c.get("intent", "unknown") for c in filtered_conversations]
        intent_counts = Counter(intents)
        
        # Sentiment analysis
        sentiments = [c.get("sentiment", "neutral") for c in filtered_conversations]
        sentiment_counts = Counter(sentiments)
        
        # Escalation rate
        escalations = [c for c in filtered_conversations if c.get("escalated", False)]
        escalation_rate = len(escalations) / len(filtered_conversations) * 100
        
        # Response time analysis
        response_times = [c.get("response_time_ms", 0) for c in filtered_conversations]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Priority distribution
        priorities = [c.get("priority_score", 5) for c in filtered_conversations]
        high_priority = len([p for p in priorities if p >= 8])
        
        analytics = {
            "time_range_days": time_range_days,
            "total_conversations": len(filtered_conversations),
            "intent_distribution": dict(intent_counts),
            "sentiment_distribution": dict(sentiment_counts),
            "escalation_rate": round(escalation_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "high_priority_count": high_priority,
            "top_issues": intent_counts.most_common(5),
            "sla_violations": self.check_sla_violations(filtered_conversations),
            "trends": self.analyze_trends(filtered_conversations)
        }
        
        return analytics
    
    def check_sla_violations(self, conversations: List[Dict]) -> List[Dict]:
        violations = []
        SLA_RESPONSE_TIME_MS = 300000  # 5 minutes in milliseconds
        
        for conv in conversations:
            response_time = conv.get("response_time_ms", 0)
            if response_time > SLA_RESPONSE_TIME_MS:
                violations.append({
                    "conversation_id": conv.get("conversation_id"),
                    "response_time_ms": response_time,
                    "violation_amount": response_time - SLA_RESPONSE_TIME_MS
                })
        
        return violations
    
    def analyze_trends(self, conversations: List[Dict]) -> Dict[str, Any]:
        # Daily trends
        daily_data = defaultdict(int)
        for conv in conversations:
            date = datetime.fromisoformat(conv["timestamp"]).date()
            daily_data[date.isoformat()] += 1
        
        # Hourly patterns
        hourly_data = defaultdict(int)
        for conv in conversations:
            hour = datetime.fromisoformat(conv["timestamp"]).hour
            hourly_data[hour] += 1
        
        return {
            "daily_volume": dict(daily_data),
            "hourly_distribution": dict(hourly_data),
            "peak_hours": [hour for hour, count in sorted(hourly_data.items(), key=lambda x: x[1], reverse=True)[:3]]
        }
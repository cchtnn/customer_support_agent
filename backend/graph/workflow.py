from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from backend.agents.intent_agent import IntentClassificationAgent
from backend.agents.sentiment_agent import SentimentPriorityAgent
from backend.agents.rag_agent import KnowledgeBaseRetrievalAgent
from backend.agents.response_agent import ResponseGenerationAgent
from backend.agents.escalation_agent import EscalationAgent
from backend.agents.analytics_agent import AnalyticsAgent
from backend.agents.qa_agent import QAComplianceAgent
import time
import uuid
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class AgentState(TypedDict):
    conversation_id: str
    query: str
    intent_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    rag_data: Dict[str, Any]
    response_data: Dict[str, Any]
    qa_data: Dict[str, Any]
    escalation_data: Dict[str, Any]
    final_response: str
    response_time_ms: float
    escalated: bool

class CustomerSupportWorkflow:
    def __init__(self):
        self.intent_agent = IntentClassificationAgent()
        self.sentiment_agent = SentimentPriorityAgent()
        self.rag_agent = KnowledgeBaseRetrievalAgent()
        self.response_agent = ResponseGenerationAgent()
        self.escalation_agent = EscalationAgent()
        self.analytics_agent = AnalyticsAgent()
        self.qa_agent = QAComplianceAgent()
        
        self.workflow = self.build_workflow()
    
    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("qa_validation", self.qa_validation)
        workflow.add_node("check_escalation", self.check_escalation)
        
        # Define edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "generate_response")
        workflow.add_edge("generate_response", "qa_validation")
        workflow.add_edge("qa_validation", "check_escalation")
        
        # Conditional routing from escalation
        workflow.add_conditional_edges(
            "check_escalation",
            self.should_escalate,
            {
                True: END,
                False: END
            }
        )
        
        return workflow.compile()
    
    async def classify_intent(self, state: AgentState) -> AgentState:
        intent_data = await self.intent_agent.classify(state["query"])
        state["intent_data"] = intent_data
        return state
    
    async def analyze_sentiment(self, state: AgentState) -> AgentState:
        sentiment_data = await self.sentiment_agent.analyze(
            state["query"], 
            state["intent_data"]["intent"]
        )
        state["sentiment_data"] = sentiment_data
        return state
    
    async def retrieve_knowledge(self, state: AgentState) -> AgentState:
        docs = await self.rag_agent.retrieve_knowledge(
            state["query"],
            state["intent_data"]["intent"]
        )
        
        rag_answer = await self.rag_agent.answer_with_context(
            state["query"],
            state["intent_data"]["intent"],
            docs
        )
        
        state["rag_data"] = rag_answer
        return state
    
    async def generate_response(self, state: AgentState) -> AgentState:
        response_data = await self.response_agent.generate_response(
            state["query"],
            state["intent_data"]["intent"],
            state["sentiment_data"],
            state["rag_data"]
        )
        state["response_data"] = response_data
        return state
    
    async def qa_validation(self, state: AgentState) -> AgentState:
        qa_data = await self.qa_agent.validate_response(
            state["query"],
            state["response_data"]["response_text"],
            state["rag_data"]["answer"]
        )
        state["qa_data"] = qa_data
        
        # If QA fails, modify response
        if not qa_data["is_compliant"]:
            state["response_data"]["response_text"] = (
                "I need to verify this information. Let me transfer you to a specialist.\n"
                f"Original response: {state['response_data']['response_text']}"
            )
            state["response_data"]["confidence"] = 0.3
        
        return state
    
    async def check_escalation(self, state: AgentState) -> AgentState:
        escalation_data = await self.escalation_agent.should_escalate(
            state["query"],
            state["intent_data"]["intent"],
            state["sentiment_data"],
            state["response_data"],
            None  # Would include conversation history in production
        )
        
        state["escalation_data"] = escalation_data
        state["escalated"] = escalation_data["should_escalate"]
        
        if escalation_data["should_escalate"]:
            state["final_response"] = (
                f"I've escalated your issue to a human agent. "
                f"Ticket ID: {escalation_data['ticket']['ticket_id']}\n"
                f"Reason: {', '.join(escalation_data['reasons'])}\n\n"
                f"{state['response_data']['response_text']}"
            )
        else:
            state["final_response"] = state["response_data"]["response_text"]
        
        return state
    
    def should_escalate(self, state: AgentState) -> bool:
        return state["escalated"]
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"process_query start: query={query[:120]}")
        
        initial_state = {
            "conversation_id": str(uuid.uuid4()),
            "query": query,
            "intent_data": {},
            "sentiment_data": {},
            "rag_data": {},
            "response_data": {},
            "qa_data": {},
            "escalation_data": {},
            "final_response": "",
            "response_time_ms": 0,
            "escalated": False
        }
        
        final_state = await self.workflow.ainvoke(initial_state)
        response_time_ms = (time.time() - start_time) * 1000
        final_state["response_time_ms"] = response_time_ms
        
        # Track for analytics
        analytics_data = {
            "conversation_id": final_state["conversation_id"],
            "query": query,
            "intent": final_state["intent_data"].get("intent"),
            "sentiment": final_state["sentiment_data"].get("sentiment"),
            "priority_score": final_state["sentiment_data"].get("priority_score"),
            "escalated": final_state["escalated"],
            "response_time_ms": response_time_ms,
            "timestamp": time.time()
        }
        self.analytics_agent.track_conversation(analytics_data)
        logger.info(f"process_query completed: conversation_id={final_state['conversation_id']} response_time_ms={response_time_ms:.2f}ms")

        return {
            "conversation_id": final_state["conversation_id"],
            "response": final_state["final_response"],
            "intent": final_state["intent_data"],
            "sentiment": final_state["sentiment_data"],
            "qa_results": final_state["qa_data"],
            "escalation": final_state["escalation_data"],
            "response_time_ms": response_time_ms,
            "confidence": final_state["response_data"].get("confidence", 0.5)
        }
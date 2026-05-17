from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json
import re
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


def parse_json_response(text: str) -> Optional[Any]:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_text = extract_json_text(text)
    if json_text:
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    return None

class QAComplianceAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=0.1
        )
        
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[\w\.-]+@[\w\.-]+\.\w+\b',  # Email
            r'\b\d{10}\b',  # Phone number
            r'\b\d{5}(?:-\d{4})?\b'  # ZIP code
        ]
    
    async def validate_response(
        self,
        query: str,
        response: str,
        rag_context: str
    ) -> Dict[str, Any]:
        
        issues = []
        
        # Check for PII leakage
        pii_leaks = self.check_pii_leakage(response)
        if pii_leaks:
            issues.append({
                "type": "pii_leakage",
                "details": f"Found potential PII: {pii_leaks}"
            })
        
        # Check for hallucinations
        hallucination_score = await self.check_hallucinations(query, response, rag_context)
        if hallucination_score > 0.3:
            issues.append({
                "type": "hallucination",
                "details": f"High hallucination score: {hallucination_score}"
            })
        
        # Check policy compliance
        policy_violations = await self.check_policy_compliance(response)
        if policy_violations:
            issues.extend(policy_violations)
        
        # Check harmful content
        harmful_score = await self.check_harmful_content(response)
        if harmful_score > 0.2:
            issues.append({
                "type": "harmful_content",
                "details": f"Harmful content score: {harmful_score}"
            })
        
        is_compliant = len(issues) == 0
        
        return {
            "is_compliant": is_compliant,
            "issues": issues,
            "hallucination_score": hallucination_score if 'hallucination_score' in locals() else 0,
            "requires_review": len(issues) > 0,
            "suggested_corrections": self.suggest_corrections(issues) if issues else []
        }
    
    def check_pii_leakage(self, text: str) -> List[str]:
        found_pii = []
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, text)
            if matches:
                found_pii.extend(matches)
        return found_pii
    
    async def check_hallucinations(self, query: str, response: str, context: str) -> float:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checking expert. Compare the response with the context.
            Rate hallucination from 0 to 1:
            0 = Response fully matches context
            0.5 = Some unsupported claims
            1 = Response completely fabricated
            
            Return only a number between 0 and 1."""),
            ("human", "Context: {context}\nQuery: {query}\nResponse: {response}\nHallucination score:")
        ])
        
        chain = prompt | self.llm
        response = await async_retry(
            lambda: chain.ainvoke({
                "context": context,
                "query": query,
                "response": response
            }),
            retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )
        
        try:
            score = float(response.content.strip())
            return min(max(score, 0), 1)
        except:
            return 0.5
    
    async def check_policy_compliance(self, response: str) -> List[Dict]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a policy compliance expert. Check if the response violates any company policies.
            
            Common violations:
            - Promising refunds without authorization
            - Making unrealistic delivery guarantees
            - Sharing internal procedures
            - Using inappropriate language
            - Making legal statements
            
            Return ONLY a valid JSON array of violations found, with no additional text.
            Example: [{{"type": "violation_type", "details": "specific issue"}}]"""),
            ("human", "Response: {response}\nViolations:")
        ])
        
        chain = prompt | self.llm
        logger.debug("QAComplianceAgent.check_policy_compliance invoking LLM")
        response_text = await async_retry(
            lambda: chain.ainvoke({"response": response}),
            retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )

        raw_text = response_text.content
        if isinstance(raw_text, bytes):
            raw_text = raw_text.decode("utf-8", errors="ignore")

        violations = parse_json_response(raw_text)
        if isinstance(violations, list):
            return violations

        logger.error(
            "Failed to parse policy compliance response, raw output: %s",
            raw_text.replace("\n", " ")[:1000]
        )
        return []
    
    async def check_harmful_content(self, response: str) -> float:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a content safety expert. Rate the response for harmful content (0-1):
            0 = Completely safe
            0.3 = Mild concerns
            0.7 = Significant issues
            1 = Dangerous/harmful content
            
            Return only a number between 0 and 1."""),
            ("human", "Response: {response}\nHarmful score:")
        ])
        
        chain = prompt | self.llm
        response_score = await async_retry(
            lambda: chain.ainvoke({"response": response}),
            retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )
        
        try:
            score = float(response_score.content.strip())
            return min(max(score, 0), 1)
        except:
            return 0
    
    def suggest_corrections(self, issues: List[Dict]) -> List[str]:
        suggestions = []
        for issue in issues:
            if issue["type"] == "pii_leakage":
                suggestions.append("Remove or mask personally identifiable information")
            elif issue["type"] == "hallucination":
                suggestions.append("Stick to information provided in the knowledge base")
            elif issue["type"] == "policy_violation":
                suggestions.append("Review company policy guidelines before responding")
        return suggestions
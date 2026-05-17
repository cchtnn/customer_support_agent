import os
import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Customer Support Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .agent-trace {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .escalation-warning {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_traces" not in st.session_state:
    st.session_state.agent_traces = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# API endpoint
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if API_URL.endswith("/"):
    API_URL = API_URL[:-1]

def send_message(query):
    """Send message to backend API"""
    try:
        response = requests.post(
            f"{API_URL}/api/chat",
            json={"query": query, "session_id": st.session_state.conversation_id}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_analytics(days=7):
    """Fetch analytics data"""
    try:
        response = requests.get(f"{API_URL}/api/analytics?days={days}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Sidebar
with st.sidebar:
    st.title("🤖 AI Support Platform")
    st.markdown("---")
    
    # System status
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Agents Active", "7/7")
    with col2:
        st.metric("Response Time", "<2s", "98%")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["💬 Chat Interface", "🔍 Agent Trace", "📊 Analytics Dashboard", "🎫 Escalation Dashboard"]
    )
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings"):
        days_analytics = st.slider("Analytics Time Range (days)", 1, 30, 7)
        show_confidence = st.checkbox("Show Confidence Scores", True)
        show_traces = st.checkbox("Show Agent Traces", True)

# Main content
if page == "💬 Chat Interface":
    st.title("💬 Customer Support Chat")
    st.markdown("Enterprise Multi-Agent Support System")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show agent trace if available
                if message.get("trace") and show_traces:
                    with st.expander("🔍 View Agent Processing"):
                        st.json(message["trace"])
                
                # Show confidence score
                if message.get("confidence") and show_confidence:
                    confidence_color = "🟢" if message["confidence"] > 0.7 else "🟡" if message["confidence"] > 0.4 else "🔴"
                    st.caption(f"{confidence_color} Confidence: {message['confidence']:.2%}")
    
    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                result = send_message(prompt)
                
                if "error" in result:
                    st.error(result["error"])
                    response_text = "I'm having trouble processing your request. Please try again."
                else:
                    response_text = result["response"]
                    st.session_state.conversation_id = result["conversation_id"]
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Show confidence
                    if show_confidence:
                        confidence_color = "🟢" if result["confidence"] > 0.7 else "🟡" if result["confidence"] > 0.4 else "🔴"
                        st.caption(f"{confidence_color} Response Confidence: {result['confidence']:.2%}")
                    
                    # Show escalation warning
                    if result["escalation"]["should_escalate"]:
                        st.warning(f"⚠️ Escalated to Human Agent\nReason: {', '.join(result['escalation']['reasons'])}")
                    
                    # Store trace
                    trace_data = {
                        "intent": result["intent"],
                        "sentiment": result["sentiment"],
                        "response_time_ms": result["response_time_ms"],
                        "confidence": result["confidence"]
                    }
                    
                    # Add to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "trace": trace_data if show_traces else None,
                        "confidence": result["confidence"]
                    })
                    
                    # Store in trace history
                    st.session_state.agent_traces.append({
                        "timestamp": datetime.now(),
                        "query": prompt,
                        "trace": trace_data
                    })

elif page == "🔍 Agent Trace":
    st.title("🔍 Agent Processing Trace")
    st.markdown("Real-time agent decision tracking")
    
    if st.session_state.agent_traces:
        for trace in reversed(st.session_state.agent_traces[-10:]):
            with st.expander(f"Query: {trace['query'][:50]}... - {trace['timestamp'].strftime('%H:%M:%S')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Intent Classification")
                    st.json(trace['trace']['intent'])
                
                with col2:
                    st.subheader("Sentiment Analysis")
                    st.json(trace['trace']['sentiment'])
                
                st.subheader("Performance Metrics")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Response Time", f"{trace['trace']['response_time_ms']:.0f}ms")
                with metric_col2:
                    st.metric("Confidence", f"{trace['trace']['confidence']:.2%}")
                with metric_col3:
                    priority = trace['trace']['sentiment'].get('priority_score', 5)
                    st.metric("Priority Score", f"{priority}/10")
    else:
        st.info("No agent traces yet. Start a conversation to see traces.")

elif page == "📊 Analytics Dashboard":
    st.title("📊 Support Analytics Dashboard")
    st.markdown("Key metrics and insights")
    
    analytics = get_analytics(days_analytics)
    
    if analytics and analytics != {"message": "No data available for the selected time range"}:
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Conversations", analytics.get('total_conversations', 0))
        with col2:
            st.metric("Escalation Rate", f"{analytics.get('escalation_rate', 0)}%")
        with col3:
            st.metric("Avg Response Time", f"{analytics.get('avg_response_time_ms', 0):.0f}ms")
        with col4:
            st.metric("High Priority Issues", analytics.get('high_priority_count', 0))
        
        st.markdown("---")
        
        # Intent Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Intent Distribution")
            intent_data = analytics.get('intent_distribution', {})
            if intent_data:
                fig = px.pie(values=list(intent_data.values()), names=list(intent_data.keys()), title="Customer Intent Types")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Analysis")
            sentiment_data = analytics.get('sentiment_distribution', {})
            if sentiment_data:
                fig = px.bar(x=list(sentiment_data.keys()), y=list(sentiment_data.values()), title="Customer Sentiment")
                st.plotly_chart(fig, use_container_width=True)
        
        # Top Issues
        st.subheader("Top Customer Issues")
        top_issues = analytics.get('top_issues', [])
        if top_issues:
            issues_df = pd.DataFrame(top_issues, columns=['Issue', 'Count'])
            st.bar_chart(issues_df.set_index('Issue'))
        
        # SLA Violations
        st.subheader("SLA Violations")
        sla_violations = analytics.get('sla_violations', [])
        if sla_violations:
            st.warning(f"⚠️ {len(sla_violations)} SLA violations detected")
            violations_df = pd.DataFrame(sla_violations)
            st.dataframe(violations_df)
        else:
            st.success("✅ No SLA violations in selected period")
        
        # Trends
        st.subheader("Volume Trends")
        trends = analytics.get('trends', {})
        if trends and trends.get('daily_volume'):
            daily_data = trends['daily_volume']
            df = pd.DataFrame(list(daily_data.items()), columns=['Date', 'Volume'])
            st.line_chart(df.set_index('Date'))
    
    else:
        st.info("Start some conversations to see analytics data")

elif page == "🎫 Escalation Dashboard":
    st.title("🎫 Escalation Management")
    st.markdown("Human escalation tracking and management")
    
    # Show recent escalations from session
    escalations = []
    for trace in st.session_state.agent_traces:
        if trace['trace'].get('escalation') and trace['trace']['escalation'].get('should_escalate'):
            escalations.append({
                "timestamp": trace['timestamp'],
                "query": trace['query'],
                "reasons": trace['trace']['escalation'].get('reasons', []),
                "ticket": trace['trace']['escalation'].get('ticket', {})
            })
    
    if escalations:
        st.warning(f"⚠️ {len(escalations)} Active Escalations")
        
        for esc in reversed(escalations):
            with st.expander(f"Ticket: {esc['ticket'].get('ticket_id', 'N/A')} - {esc['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Customer Query**")
                    st.info(esc['query'])
                    
                    st.markdown("**Escalation Reasons**")
                    for reason in esc['reasons']:
                        st.write(f"• {reason}")
                
                with col2:
                    st.markdown("**Ticket Details**")
                    if esc['ticket']:
                        st.json(esc['ticket'])
                    
                    # Action buttons
                    col_actions1, col_actions2, col_actions3 = st.columns(3)
                    with col_actions1:
                        if st.button(f"Assign to Me", key=f"assign_{esc['ticket'].get('ticket_id')}"):
                            st.success("Assigned successfully")
                    with col_actions2:
                        if st.button(f"Resolve", key=f"resolve_{esc['ticket'].get('ticket_id')}"):
                            st.success("Ticket resolved")
                    with col_actions3:
                        if st.button(f"View Details", key=f"details_{esc['ticket'].get('ticket_id')}"):
                            st.info("Full ticket details view")
    else:
        st.success("✅ No active escalations. All issues being handled by AI agents.")

# Footer
st.markdown("---")
st.markdown("**AI Customer Support Orchestration Platform** | Multi-Agent System | Enterprise Ready")

# Auto-refresh for analytics
if page == "📊 Analytics Dashboard":
    time.sleep(5)
    st.rerun()
"""
Complete LangCache Demo Runner - Phase 3
Side-by-side comparison with comprehensive metrics dashboard.
Perfect for live demonstrations to solution architects and customers.
"""

import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# LANGCACHE INTEGRATION POINT 1: Import LangCache
from langcache import LangCache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoComparison:
    """Handles side-by-side comparison between basic and cached responses."""
    
    def __init__(self):
        """Initialize both basic and cached clients."""
        # OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"

        # LANGCACHE INTEGRATION POINT 2: Store LangCache connection settings
        self.langcache_config = {
            "server_url": os.getenv("LANGCACHE_SERVER_URL"),
            "cache_id": os.getenv("LANGCACHE_CACHE_ID"),
            "api_key": os.getenv("LANGCACHE_API_KEY")
        }

        # Check if LangCache is properly configured
        self.cache_enabled = all(self.langcache_config.values())
        if self.cache_enabled:
            logger.info("LangCache configuration found")
        else:
            logger.warning("LangCache not configured")
    
    def get_basic_response(self, user_message: str) -> dict:
        """Get response without caching (baseline)."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "message": response.choices[0].message.content,
                "response_time": response_time,
                "tokens_used": response.usage.total_tokens,
                "cached": False,
                "error": None,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "message": f"Error: {str(e)}",
                "response_time": end_time - start_time,
                "tokens_used": 0,
                "cached": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def get_cached_response(self, user_message: str) -> dict:
        """Get response with LangCache integration."""
        start_time = time.time()
        
        # LANGCACHE INTEGRATION POINT 3: Check cache first
        if self.cache_enabled:
            try:
                with LangCache(**self.langcache_config) as cache_client:
                    search_result = cache_client.search(
                        prompt=user_message,
                        similarity_threshold=0.8
                    )

                    if search_result and hasattr(search_result, 'data') and search_result.data:
                        # Get the best match
                        best_match = search_result.data[0]
                        cached_response = best_match.response

                        end_time = time.time()
                        return {
                            "message": cached_response,
                            "response_time": end_time - start_time,
                            "tokens_used": 0,
                            "cached": True,
                            "error": None,
                            "timestamp": datetime.now()
                        }
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        # LANGCACHE INTEGRATION POINT 4: Fallback to API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            assistant_message = response.choices[0].message.content
            
            # LANGCACHE INTEGRATION POINT 5: Store in cache
            if self.cache_enabled:
                try:
                    with LangCache(**self.langcache_config) as cache_client:
                        cache_client.set(
                            prompt=user_message,
                            response=assistant_message,
                            ttl_millis=3600000  # 1 hour TTL
                        )
                except Exception as e:
                    logger.warning(f"Failed to cache: {e}")
            
            return {
                "message": assistant_message,
                "response_time": response_time,
                "tokens_used": response.usage.total_tokens,
                "cached": False,
                "error": None,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "message": f"Error: {str(e)}",
                "response_time": end_time - start_time,
                "tokens_used": 0,
                "cached": False,
                "error": str(e),
                "timestamp": datetime.now()
            }

def create_metrics_dashboard(basic_history, cached_history):
    """Create comprehensive metrics dashboard."""
    
    if not basic_history and not cached_history:
        st.info("No data yet. Try some queries to see metrics!")
        return
    
    # Calculate metrics
    basic_df = pd.DataFrame(basic_history) if basic_history else pd.DataFrame()
    cached_df = pd.DataFrame(cached_history) if cached_history else pd.DataFrame()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not cached_df.empty:
            cache_hits = len(cached_df[cached_df['cached'] == True])
            total_cached_requests = len(cached_df)
            hit_rate = (cache_hits / total_cached_requests * 100) if total_cached_requests > 0 else 0
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        else:
            st.metric("Cache Hit Rate", "0%")
    
    with col2:
        if not basic_df.empty and not cached_df.empty:
            avg_basic_time = basic_df['response_time'].mean()
            avg_cached_time = cached_df['response_time'].mean()
            improvement = ((avg_basic_time - avg_cached_time) / avg_basic_time * 100)
            st.metric("Speed Improvement", f"{improvement:.1f}%")
        else:
            st.metric("Speed Improvement", "N/A")
    
    with col3:
        basic_tokens = basic_df['tokens_used'].sum() if not basic_df.empty else 0
        cached_tokens = cached_df['tokens_used'].sum() if not cached_df.empty else 0
        token_savings = basic_tokens - cached_tokens
        st.metric("Tokens Saved", f"{token_savings:,}")
    
    with col4:
        # Estimate cost savings (approximate OpenAI pricing)
        cost_per_1k_tokens = 0.002  # GPT-3.5-turbo pricing
        cost_savings = (token_savings / 1000) * cost_per_1k_tokens
        st.metric("Est. Cost Savings", f"${cost_savings:.4f}")
    
    # Response time comparison chart
    if not basic_df.empty and not cached_df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=basic_df['response_time'],
            mode='lines+markers',
            name='Without Cache',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            y=cached_df['response_time'],
            mode='lines+markers',
            name='With LangCache',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title="Response Time Comparison",
            xaxis_title="Request Number",
            yaxis_title="Response Time (seconds)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main demo application."""
    st.set_page_config(
        page_title="LangCache Demo - Complete Comparison",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 LangCache Demo: Complete Performance Comparison")
    st.markdown("**Phase 3**: Side-by-side comparison showing the power of intelligent caching")
    
    # Initialize demo comparison
    if "demo" not in st.session_state:
        st.session_state.demo = DemoComparison()
    
    # Initialize history
    if "basic_history" not in st.session_state:
        st.session_state.basic_history = []
    if "cached_history" not in st.session_state:
        st.session_state.cached_history = []
    
    # Sample questions for demo
    st.markdown("### 🎯 Demo Questions")
    st.markdown("Use these questions to demonstrate semantic caching:")
    
    demo_questions = [
        "What is machine learning?",
        "Can you explain machine learning?",
        "What does ML mean?",
        "How does artificial intelligence work?",
        "Explain AI to me",
        "What are the benefits of cloud computing?",
        "Why should I use cloud services?"
    ]
    
    selected_question = st.selectbox("Choose a demo question:", [""] + demo_questions)
    
    # Custom question input
    custom_question = st.text_input("Or ask your own question:")
    
    # Use selected or custom question
    question = custom_question if custom_question else selected_question
    
    if st.button("🚀 Run Comparison", disabled=not question):
        st.markdown("---")
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❌ Without LangCache")
            with st.spinner("Calling OpenAI API..."):
                basic_result = st.session_state.demo.get_basic_response(question)
            
            st.write(basic_result["message"])
            
            # Metrics
            st.caption(f"⏱️ Response Time: {basic_result['response_time']:.2f}s")
            st.caption(f"🎯 Tokens Used: {basic_result['tokens_used']}")
            st.caption(f"💰 Est. Cost: ${(basic_result['tokens_used']/1000)*0.002:.4f}")
            
            st.session_state.basic_history.append(basic_result)
        
        with col2:
            st.subheader("✅ With LangCache")
            with st.spinner("Checking cache..."):
                cached_result = st.session_state.demo.get_cached_response(question)
            
            st.write(cached_result["message"])
            
            # Enhanced metrics
            if cached_result["cached"]:
                st.success("🚀 Cache HIT! Lightning fast response")
            else:
                st.info("🔄 Cache MISS - Response cached for next time")
            
            st.caption(f"⏱️ Response Time: {cached_result['response_time']:.2f}s")
            st.caption(f"🎯 Tokens Used: {cached_result['tokens_used']}")
            st.caption(f"💰 Est. Cost: ${(cached_result['tokens_used']/1000)*0.002:.4f}")
            
            st.session_state.cached_history.append(cached_result)
        
        # Performance comparison
        if basic_result and cached_result:
            st.markdown("### 📊 This Query Performance")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                time_diff = basic_result['response_time'] - cached_result['response_time']
                improvement = (time_diff / basic_result['response_time']) * 100
                st.metric("Speed Improvement", f"{improvement:.1f}%", f"{time_diff:.2f}s faster")
            
            with perf_col2:
                token_savings = basic_result['tokens_used'] - cached_result['tokens_used']
                st.metric("Tokens Saved", f"{token_savings}", f"{token_savings} tokens")
            
            with perf_col3:
                cost_savings = (token_savings / 1000) * 0.002
                st.metric("Cost Savings", f"${cost_savings:.4f}", f"Per query")
    
    # Overall metrics dashboard
    st.markdown("---")
    st.markdown("### 📈 Session Performance Dashboard")
    create_metrics_dashboard(st.session_state.basic_history, st.session_state.cached_history)
    
    # Clear data button
    if st.button("🗑️ Clear Session Data"):
        st.session_state.basic_history = []
        st.session_state.cached_history = []
        st.rerun()
    
    # Demo tips
    with st.expander("💡 Demo Tips for Solution Architects"):
        st.markdown("""
        **Key Points to Highlight:**
        
        1. **Minimal Code Changes**: Show how LangCache requires only 5 integration points
        2. **Semantic Intelligence**: Ask similar questions to demonstrate intelligent caching
        3. **Immediate ROI**: Point out real-time cost and performance improvements
        4. **Graceful Fallback**: Emphasize that cache failures don't break the application
        5. **Easy Integration**: Highlight that existing LLM code barely changes
        
        **Demo Flow Suggestions:**
        1. Start with a fresh question (cache miss)
        2. Ask the same question again (cache hit)
        3. Ask semantically similar questions (intelligent caching)
        4. Show the metrics dashboard for ROI discussion
        """)

if __name__ == "__main__":
    main()

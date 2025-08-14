"""
LangCache Integrated Chatbot - Phase 2
Demonstrates exact integration points for adding LangCache to existing LLM applications.
This shows the minimal code changes needed to add intelligent caching.
"""

import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import logging

# LANGCACHE INTEGRATION POINT 1: Import LangCache
from langcache import LangCache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangCacheChatbot:
    """Chatbot with LangCache integration for intelligent semantic caching."""
    
    def __init__(self):
        """Initialize the chatbot with OpenAI client."""
        # Original OpenAI client initialization
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
            logger.warning("LangCache not configured. Running without cache.")
        
    def get_response(self, user_message: str) -> dict:
        """
        Get response with LangCache integration.
        
        Args:
            user_message: User's input message
            
        Returns:
            dict: Response data including message, tokens, timing, and cache status
        """
        start_time = time.time()
        
        # LANGCACHE INTEGRATION POINT 3: Check cache first
        if self.cache_enabled:
            try:
                with LangCache(**self.langcache_config) as cache_client:
                    search_result = cache_client.search(
                        prompt=user_message,
                        similarity_threshold=0.8
                    )

                    # Debug logging
                    logger.info(f"Search result: {search_result}")
                    if search_result:
                        logger.info(f"Has entries attr: {hasattr(search_result, 'entries')}")
                        if hasattr(search_result, 'entries'):
                            logger.info(f"Entries: {search_result.entries}")
                            logger.info(f"Entries length: {len(search_result.entries) if search_result.entries else 0}")

                    if search_result and hasattr(search_result, 'data') and search_result.data:
                        # Get the best match
                        best_match = search_result.data[0]
                        cached_response = best_match.response

                        end_time = time.time()
                        response_time = end_time - start_time

                        logger.info(f"Cache HIT for query: '{user_message[:50]}...'")

                        # Estimate tokens saved based on response length (rough approximation: 1 token ≈ 4 characters)
                        estimated_tokens_saved = len(cached_response) // 4

                        return {
                            "message": cached_response,
                            "response_time": response_time,
                            "tokens_used": 0,  # No tokens used for cached responses
                            "tokens_saved": estimated_tokens_saved,  # Estimated based on response length
                            "cached": True,
                            "error": None
                        }
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}. Falling back to API.")
        
        # Original API call (unchanged from basic version)
        try:
            # Make API call to OpenAI
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
            
            # Extract response data (same as before)
            assistant_message = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # LANGCACHE INTEGRATION POINT 4: Store in cache
            if self.cache_enabled:
                try:
                    with LangCache(**self.langcache_config) as cache_client:
                        cache_client.set(
                            prompt=user_message,
                            response=assistant_message,
                            ttl_millis=3600000  # 1 hour TTL
                        )
                    logger.info(f"Cached response for query: '{user_message[:50]}...'")
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            logger.info(f"OpenAI API call completed in {response_time:.2f}s, {tokens_used} tokens")
            
            return {
                "message": assistant_message,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "cached": False,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.error(f"OpenAI API error: {str(e)}")
            
            return {
                "message": f"Sorry, I encountered an error: {str(e)}",
                "response_time": response_time,
                "tokens_used": 0,
                "cached": False,
                "error": str(e)
            }

def main():
    """Main Streamlit application with LangCache integration."""
    st.set_page_config(
        page_title="LangCache Integrated Chatbot - Phase 2",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ LangCache Integrated Chatbot")
    st.markdown("**Phase 2**: Same chatbot with intelligent semantic caching")
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = LangCacheChatbot()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize metrics
    if "total_requests" not in st.session_state:
        st.session_state.total_requests = 0
        st.session_state.total_tokens_used = 0
        st.session_state.total_tokens_saved = 0
        st.session_state.total_time = 0
        st.session_state.cache_hits = 0
    
    # Sidebar with enhanced metrics
    with st.sidebar:
        st.header("📊 Session Metrics")
        st.metric("Total Requests", st.session_state.total_requests)
        st.metric("Cache Hits", st.session_state.cache_hits)

        if st.session_state.total_requests > 0:
            hit_rate = (st.session_state.cache_hits / st.session_state.total_requests) * 100
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")

        st.metric("Output Tokens Used", f"{st.session_state.total_tokens_used:,}")
        st.metric("Output Tokens Saved", f"{st.session_state.total_tokens_saved:,}")

        # Calculate cost savings ($10 per million tokens)
        cost_per_million = 10.0
        tokens_used_cost = (st.session_state.total_tokens_used / 1_000_000) * cost_per_million
        tokens_saved_cost = (st.session_state.total_tokens_saved / 1_000_000) * cost_per_million

        st.metric("Cost Spent", f"${tokens_used_cost:.4f}")
        st.metric("Cost Saved", f"${tokens_saved_cost:.4f}")

        if st.session_state.total_requests > 0:
            avg_time = st.session_state.total_time / st.session_state.total_requests
            st.metric("Avg Response Time", f"{avg_time:.2f}s")

        st.markdown("---")
        st.markdown("**LangCache Status**: " +
                   ("✅ Enabled" if st.session_state.chatbot.cache_enabled else "❌ Disabled"))

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.total_requests = 0
            st.session_state.total_tokens_used = 0
            st.session_state.total_tokens_saved = 0
            st.session_state.total_time = 0
            st.session_state.cache_hits = 0
            st.rerun()
    
    # Demo instructions for testing semantic caching
    st.markdown("### 🧪 Demo: Test Semantic Caching")
    st.markdown("**To see LangCache in action, try this sequence:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Step 1: First Question**")
        if st.button("Q1: What is machine learning?", key="demo_q1"):
            st.session_state.sample_question = "What is machine learning?"
        st.caption("This will be a cache MISS")

    with col2:
        st.markdown("**Step 2: Similar Question**")
        if st.button("Q2: Can you explain ML?", key="demo_q2"):
            st.session_state.sample_question = "Can you explain machine learning?"
        st.caption("This should be a cache HIT! 🚀")

    with col3:
        st.markdown("**Step 3: Another Variation**")
        if st.button("Q3: What does ML mean?", key="demo_q3"):
            st.session_state.sample_question = "What does ML mean?"
        st.caption("This should also be a cache HIT! ⚡")

    st.markdown("**OR write your own question in the chat below** 👇")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show enhanced metrics for assistant messages
            if message["role"] == "assistant" and "metrics" in message:
                metrics = message["metrics"]
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption(f"⏱️ {metrics['response_time']:.2f}s")
                with col2:
                    if metrics["cached"]:
                        tokens_saved = metrics.get('tokens_saved', 100)
                        st.caption(f"💰 {tokens_saved} tokens saved")
                    else:
                        st.caption(f"🎯 {metrics['tokens_used']} tokens used")
                with col3:
                    if metrics.get("error"):
                        st.caption("❌ Error")
                    elif metrics["cached"]:
                        st.caption("🚀 Cache HIT")
                    else:
                        st.caption("🔄 API Call")
    
    # Handle sample question selection
    if hasattr(st.session_state, 'sample_question'):
        prompt = st.session_state.sample_question
        del st.session_state.sample_question
    else:
        prompt = st.chat_input("Ask me anything...")
    
    # Process user input
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = st.session_state.chatbot.get_response(prompt)
            
            st.markdown(response_data["message"])
            
            # Display enhanced metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ {response_data['response_time']:.2f}s")
            with col2:
                if response_data["cached"]:
                    tokens_saved = response_data.get('tokens_saved', 100)
                    st.caption(f"💰 {tokens_saved} tokens saved")
                else:
                    st.caption(f"🎯 {response_data['tokens_used']} tokens used")
            with col3:
                if response_data.get("error"):
                    st.caption("❌ Error")
                elif response_data["cached"]:
                    st.caption("🚀 Cache HIT")
                else:
                    st.caption("🔄 API Call")
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_data["message"],
            "metrics": response_data
        })
        
        # Update session metrics
        st.session_state.total_requests += 1
        st.session_state.total_time += response_data["response_time"]

        if response_data["cached"]:
            st.session_state.cache_hits += 1
            # Use actual tokens saved from the cached response
            tokens_saved = response_data.get('tokens_saved', 100)
            st.session_state.total_tokens_saved += tokens_saved
        else:
            st.session_state.total_tokens_used += response_data["tokens_used"]
        
        st.rerun()

if __name__ == "__main__":
    main()

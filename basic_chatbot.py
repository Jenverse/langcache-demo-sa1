"""
Basic LLM Chatbot - Phase 1
A simple chatbot using OpenAI API without caching.
This demonstrates the baseline before LangCache integration.
"""

import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicChatbot:
    """Basic chatbot using OpenAI API without any caching."""
    
    def __init__(self):
        """Initialize the chatbot with OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"
        
    def get_response(self, user_message: str) -> dict:
        """
        Get response from OpenAI API.
        
        Args:
            user_message: User's input message
            
        Returns:
            dict: Response data including message, tokens, and timing
        """
        start_time = time.time()
        
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
            
            # Extract response data
            assistant_message = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            logger.info(f"OpenAI API call completed in {response_time:.2f}s, {tokens_used} tokens")
            
            return {
                "message": assistant_message,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "cached": False,  # Always False for basic chatbot
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
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Basic LLM Chatbot - Phase 1",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Basic LLM Chatbot (Without Caching)")
    st.markdown("**Phase 1**: Baseline chatbot using OpenAI API directly")
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = BasicChatbot()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize metrics
    if "total_requests" not in st.session_state:
        st.session_state.total_requests = 0
        st.session_state.total_tokens = 0
        st.session_state.total_time = 0
    
    # Sidebar with metrics
    with st.sidebar:
        st.header("📊 Session Metrics")
        st.metric("Total Requests", st.session_state.total_requests)
        st.metric("Total Tokens Used", st.session_state.total_tokens)
        
        if st.session_state.total_requests > 0:
            avg_time = st.session_state.total_time / st.session_state.total_requests
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        st.markdown("---")
        st.markdown("**Note**: This is the baseline without caching. All requests go directly to OpenAI API.")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.total_requests = 0
            st.session_state.total_tokens = 0
            st.session_state.total_time = 0
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metrics for assistant messages
            if message["role"] == "assistant" and "metrics" in message:
                metrics = message["metrics"]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.caption(f"⏱️ {metrics['response_time']:.2f}s")
                with col2:
                    st.caption(f"🎯 {metrics['tokens_used']} tokens")
                with col3:
                    if metrics.get("error"):
                        st.caption("❌ Error")
                    else:
                        st.caption("✅ Direct API")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
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
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ {response_data['response_time']:.2f}s")
            with col2:
                st.caption(f"🎯 {response_data['tokens_used']} tokens")
            with col3:
                if response_data.get("error"):
                    st.caption("❌ Error")
                else:
                    st.caption("✅ Direct API")
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_data["message"],
            "metrics": response_data
        })
        
        # Update session metrics
        st.session_state.total_requests += 1
        st.session_state.total_tokens += response_data["tokens_used"]
        st.session_state.total_time += response_data["response_time"]
        
        st.rerun()

if __name__ == "__main__":
    main()

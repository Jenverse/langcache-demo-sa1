# LangCache Integration Demo for Solution Architects

This repository is focused on **integrating LangCache with your LLM applications** to add intelligent semantic caching, reduce costs, and improve response times.

## 🎯 Purpose

This demo shows solution architects exactly how to integrate the LangCache SDK into existing LLM applications with minimal code changes. Perfect for customer presentations and hands-on integration training.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jenverse/langcache-demo-sa1.git
   cd langcache-demo-sa1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Run the demo**
   ```bash
   # Basic chatbot (no caching)
   streamlit run basic_chatbot.py
   
   # LangCache integrated chatbot
   streamlit run langcache_chatbot.py
   
   # Side-by-side comparison demo
   streamlit run demo_runner.py
   ```

## 📁 Demo Components

- **`basic_chatbot.py`** - Baseline LLM chatbot without caching
- **`langcache_chatbot.py`** - Same chatbot with LangCache integration
- **`demo_runner.py`** - Side-by-side comparison showing cache performance
- **`INTEGRATION_GUIDE.md`** - Complete step-by-step integration instructions

---

# 🚀 LangCache Integration Guide

> **Transform your LLM app with intelligent caching in just 4 simple steps**

[![PyPI version](https://badge.fury.io/py/langcache.svg)](https://pypi.org/project/langcache/)
[![Integration Time](https://img.shields.io/badge/Integration%20Time-15%20minutes-brightgreen)](https://pypi.org/project/langcache/)
[![Code Changes](https://img.shields.io/badge/Code%20Changes-~36%20lines-blue)](https://pypi.org/project/langcache/)

**LangCache SDK Documentation**: https://pypi.org/project/langcache/

---

## 🎯 **Why LangCache?**

```diff
- Expensive LLM API calls for similar queries
- Slow response times for your users  
- High output token consumption costs
+ ✅ Instant responses for cached queries
+ ✅ Reduce API costs by up to 90%
+ ✅ Smart semantic similarity matching
```

## ⚡ **Integration Overview**

**How it works:**
1. **Cache Hit** → Return cached response instantly (no LLM API call, no output tokens used)
2. **Cache Miss** → Call OpenAI LLM API → Store response in cache for future use

| Step | What You'll Add | Lines Added |
|------|----------------|-------------|
| 🔧 Dependencies | 1 import line | +1 |
| 🔐 Environment | 3 config variables | +3 |
| ⚙️ Setup | 5 configuration lines | +5 |
| 🚀 Cache Integration | Complete cache lookup + storage around OpenAI LLM API calls | +27 |

**Total: 36 lines of code change*

---

## 🏗️ **4-Step Integration Process**

### **Step 1** 🔧 Add Dependencies

<details>
<summary><strong>📦 Click to expand - Dependencies & Imports</strong></summary>

**Add to `requirements.txt`:**
```txt
langcache>=0.0.1
```

**Add to your Python file:**
```python
# Your existing imports (keep these)
import os
from openai import OpenAI
from dotenv import load_dotenv

# ➕ Add this single line
from langcache import LangCache
```

✅ **Result**: `+1 line` added to your imports

</details>

### **Step 2** 🔐 Environment Configuration

<details>
<summary><strong>🌍 Click to expand - Environment Variables</strong></summary>

**Add to your `.env` file:**
```env
# 🟢 ADD: LangCache Configuration
LANGCACHE_SERVER_URL=https://api.example.com
LANGCACHE_CACHE_ID=your_cache_id_here
LANGCACHE_API_KEY=your_langcache_api_key_here
```

**Delta: +3 lines**

</details>

### **Step 3** ⚙️ Configuration Setup

<details>
<summary><strong>🔨 Click to expand - Configuration Setup</strong></summary>

**BEFORE - Original initialization:**
```python
class MyChatbot:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"
```

**AFTER - With LangCache configuration:**
```python
class MyChatbot:
    def __init__(self):
        # Your existing OpenAI setup (unchanged)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"
        
        # <<<< 🟢 INSERT LANGCACHE CONFIG HERE >>>>
        self.langcache_config = {
            "server_url": os.getenv("LANGCACHE_SERVER_URL"),
            "cache_id": os.getenv("LANGCACHE_CACHE_ID"),
            "api_key": os.getenv("LANGCACHE_API_KEY")
        }
        self.cache_enabled = all(self.langcache_config.values())
        # <<<< END LANGCACHE CONFIG INSERTION >>>>
```

**Delta: +5 lines**

</details>

### **Step 4** 🚀 Cache Lookup for Cache Hit + Cache Storage on Cache Miss

<details>
<summary><strong>⚡ Click to expand - Complete Cache Integration</strong></summary>

**Location**: Your main response generation method - wrap your OpenAI LLM API calls with caching

**Your existing function with complete LangCache integration:**
```python
# This is where you normally make a call to OpenAI LLM API
# But before making that call, you add cache lookup first
def get_response(self, user_message: str) -> dict:
    start_time = time.time()

    # <<<< 🟢 INSERT CACHE LOOKUP HERE >>>>
    if self.cache_enabled:
        try:
            with LangCache(**self.langcache_config) as cache_client:
                search_result = cache_client.search(
                    prompt=user_message,
                    similarity_threshold=0.8
                )

                # Check if we found a cache hit
                if search_result and hasattr(search_result, 'data') and search_result.data:
                    # CACHE HIT! Get the cached response
                    best_match = search_result.data[0]
                    cached_response = best_match.response

                    end_time = time.time()
                    return {
                        "message": cached_response,
                        "response_time": end_time - start_time,
                        "tokens_used": 0,
                        "cached": True
                    }
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}. Falling back to OpenAI LLM API.")
    # <<<< END CACHE LOOKUP INSERTION >>>>

    # Your existing OpenAI LLM API call (unchanged)
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
    assistant_message = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    # <<<< 🟢 INSERT CACHE STORAGE HERE >>>>
    # This code will only be executed if there was a CACHE MISS above
    if self.cache_enabled:
        try:
            with LangCache(**self.langcache_config) as cache_client:
                cache_client.set(
                    prompt=user_message,
                    response=assistant_message,
                    ttl_millis=2592000000  # 30 days TTL
                )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    # <<<< END CACHE STORAGE INSERTION >>>>

    # This is unchanged
    return {
        "message": assistant_message,
        "response_time": end_time - start_time,
        "tokens_used": tokens_used,
        "cached": False
    }
```

**Delta: +27 lines**

</details>

---

## 🎉 **You're Done!**

### 📊 **What You've Achieved**

```
✅ Smart semantic caching integrated
✅ Zero changes to existing OpenAI calls
✅ Automatic cost reduction
✅ Faster response times
✅ Graceful fallback if cache fails
```

### 📈 **Expected Results**

| Metric | Before LangCache | After LangCache |
|--------|------------------|-----------------|
| **Response Time** | 2-5 seconds | 50-200ms (cache hits) |
| **API Costs** | $100/month | $10-30/month |
| **Cache Hit Rate** | 0% | 60-90% typical |

---

## 🆘 **Need Help?**

- 📚 **Full Documentation**: [LangCache PyPI](https://pypi.org/project/langcache/)
- 🐛 **Issues**: Open a GitHub issue
- 💬 **Questions**: Check our discussions

---

## 🔗 **Quick Links**

- [📦 PyPI Package](https://pypi.org/project/langcache/)
- [📖 API Documentation](https://pypi.org/project/langcache/)
- [🎯 Configuration Guide](https://pypi.org/project/langcache/)

---

*Made with ❤️ for developers who want blazing fast LLM apps*

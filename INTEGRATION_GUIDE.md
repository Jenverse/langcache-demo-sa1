# 🚀 LangCache Integration Guide

> **Transform your LLM app with intelligent caching in just 5 simple steps**


**LangCache SDK Documentation**: https://pypi.org/project/langcache/

---

## 🎯 **Why LangCache?**

```diff
- Expensive API calls for similar queries
- Slow response times for your users
- High token consumption costs
+ ✅ Instant responses for cached queries
+ ✅ Reduce API costs by up to 90%
+ ✅ Smart semantic similarity matching
```

## ⚡ **Integration Overview**

**How it works:**
1. **Cache Hit** → Return cached response instantly (no LLM API call, no output tokens used)
2. **Cache Miss** → Call OpenAI LLM API → Store response in cache for future use

## � Code Changes 

| Step | What You'll Add | Lines Added |
|------|----------------|-------------|
| 🔧 Dependencies | 1 import line | +1 |
| 🔐 Environment | 3 config variables | +3 |
| ⚙️ Setup | 5 configuration lines | +5 |
| � Cache Integration | Complete cache lookup + storage around OpenAI LLM API calls | +27 |

**Total: 36 lines of code change*

---

## �📋 Detailed Integration Steps

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

### Step 2: Environment Configuration

**File to modify**: `.env` file

**Add these LangCache service variables:**
```env
# 🟢 ADD: LangCache Configuration
LANGCACHE_SERVER_URL=https://api.example.com
LANGCACHE_CACHE_ID=your_cache_id_here
LANGCACHE_API_KEY=your_langcache_api_key_here
```

**Delta: +3 lines**

### Step 3: Configuration Setup

**Location**: In your class `__init__` method or application startup

**Your existing class with LangCache configuration:**
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

### Step 4: Cache Lookup for Cache Hit + Cache Storage on Cache Miss

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





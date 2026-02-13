import asyncio
import aiofiles
import time

class RateLimiter:
    def __init__(self, max_requests_per_minute, max_tokens_per_minute):
        self.max_requests = max_requests_per_minute
        self.max_tokens = max_tokens_per_minute
        self.request_times = []
        self.token_usage = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self, estimated_tokens=1000):
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
            
            if len(self.request_times) >= self.max_requests:
                sleep_time = 60 - (now - self.request_times[0]) + 2 # add buffer
                if sleep_time > 0:
                    print(f" Request rate limit: waiting {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
                    # Refresh after sleep
                    now = time.time()
                    minute_ago = now - 60
                    self.request_times = [t for t in self.request_times if t > minute_ago]
            
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.max_tokens:
                sleep_time = 60 - (now - self.token_usage[0][0]) + 2 # add buffer
                if sleep_time > 0:
                    print(f" Token rate limit: waiting {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
                    # Refresh after sleep
                    now = time.time()
                    minute_ago = now - 60
                    self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
            
            
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))
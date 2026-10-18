import os

# Gunicorn configuration optimized for Render deployment
# 1. Port binding for Render dynamic PORT
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# 2. Worker count: exactly 1 worker to prevent memory duplication on Render Free tier (512MB RAM)
workers = 1

# 3. Threads: 2 lightweight threads for handling concurrent non-blocking I/O
threads = 2

# 4. Timeout: 120 seconds to eliminate HTTP 502 Bad Gateway timeouts from cold starts or heavy images
timeout = 120

# 5. Keepalive connection
keepalive = 5

# 6. Preload application so PyTorch model and weights are initialized and warmed up before binding port
preload_app = True

# 7. Worker class
worker_class = "gthread"

# 8. Memory leak prevention: periodically recycle worker
max_requests = 1000
max_requests_jitter = 50

# 9. Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

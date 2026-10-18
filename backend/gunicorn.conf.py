import os

# Gunicorn configuration optimized for Render deployment
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 2
timeout = 120
keepalive = 5
preload_app = True
worker_class = "gthread"
max_requests = 1000
max_requests_jitter = 50
loglevel = "info"
accesslog = "-"
errorlog = "-"

import time

def log(message: str):
    """Simple logger that prefixes messages with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

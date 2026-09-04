import psutil, os

p = psutil.Process(os.getpid())
print(f"RSS {p.memory_info().rss / 1e9:.2f} GB")

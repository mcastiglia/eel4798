import time
import psutil

def profile_block(func, *args, label="", **kwargs):
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2
    cpu_before = psutil.cpu_percent(interval=None)
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed_time = time.perf_counter() - start_time
    mem_after = process.memory_info().rss / 1024**2
    cpu_after = psutil.cpu_percent(interval=None)

    header = f"\n--- Resource Usage: {label} ---"

    print(header)
    print(f"CPU usage delta: {cpu_after - cpu_before:.2f}%")
    print(f"Memory usage delta: {mem_after - mem_before:.2f} MB")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print("-" * len(header) + "\n")
    return result

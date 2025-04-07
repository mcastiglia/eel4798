import time
import psutil
import timeit
import asyncio
import statistics
from functools import wraps
import concurrent.futures

def profile_block(func, *args, label="", sample_interval=0.1, **kwargs):
    """Profile a function's execution with detailed resource usage.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to func
        label: Label for the profiling report
        sample_interval: Time interval for resource sampling in seconds
        **kwargs: Keyword arguments to pass to func
    """
    process = psutil.Process()
    
    # Lists to store resource samples
    cpu_samples = []
    mem_samples = []
    stop_sampling = False
    
    async def sample_resources():
        while not stop_sampling:
            cpu_samples.append(psutil.cpu_percent(interval=None))
            mem_samples.append(process.memory_info().rss / 1024**2)
            await asyncio.sleep(sample_interval)
    
    # Use timeit for accurate timing
    start_mem = process.memory_info().rss / 1024**2
    
    # Start resource sampling in a separate thread
    loop = asyncio.new_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        sampling_future = executor.submit(
            lambda: loop.run_until_complete(sample_resources())
        )
        
        # Time and execute the function
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        execution_time = timeit.default_timer() - start_time
        
        # Stop the sampling
        stop_sampling = True
        loop.stop()
    
    end_mem = process.memory_info().rss / 1024**2
    
    # Generate report
    header = f"\n--- Resource Usage: {label} ---"
    print(header)
    
    # Time information
    print(f"Execution time: {execution_time:.6f} seconds")
    
    # Memory information
    if mem_samples:
        print(f"Memory usage:")
        print(f"  Initial: {start_mem:.2f} MB")
        print(f"  Final: {end_mem:.2f} MB")
        print(f"  Peak: {max(mem_samples):.2f} MB")
        print(f"  Delta: {end_mem - start_mem:.2f} MB")
    else:
        print(f"Memory usage delta: {end_mem - start_mem:.2f} MB")
    
    # CPU information
    if cpu_samples:
        print(f"CPU usage:")
        print(f"  Average: {statistics.mean(cpu_samples):.2f}%")
        print(f"  Peak: {max(cpu_samples):.2f}%")
    else:
        print(f"CPU usage: {psutil.cpu_percent():.2f}%")
        
    print("-" * len(header) + "\n")
    
    return result


def profile(label="", sample_interval=0.1):
    """Decorator for profiling functions.
    
    Args:
        label: Label for the profiling report
        sample_interval: Time interval for resource sampling in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return profile_block(func, *args, label=label or func.__name__, 
                              sample_interval=sample_interval, **kwargs)
        return wrapper
    return decorator

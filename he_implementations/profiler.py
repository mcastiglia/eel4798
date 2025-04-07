import time
import psutil
import timeit
import asyncio
import statistics
from functools import wraps
import concurrent.futures
import os

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
    disk_read_samples = []
    disk_write_samples = []
    stop_sampling = False
    
    # Get initial disk counters
    initial_disk_io = process.io_counters() if hasattr(process, 'io_counters') else None
    
    async def sample_resources():
        while not stop_sampling:
            cpu_samples.append(psutil.cpu_percent(interval=None))
            mem_samples.append(process.memory_info().rss / 1024**2)
            
            # Sample disk I/O if available
            if hasattr(process, 'io_counters'):
                io_counters = process.io_counters()
                disk_read_samples.append(io_counters.read_bytes)
                disk_write_samples.append(io_counters.write_bytes)
                
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
    final_disk_io = process.io_counters() if hasattr(process, 'io_counters') else None
    
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
    
    # Disk I/O information
    if initial_disk_io and final_disk_io:
        read_bytes = final_disk_io.read_bytes - initial_disk_io.read_bytes
        write_bytes = final_disk_io.write_bytes - initial_disk_io.write_bytes
        
        print(f"Disk I/O:")
        print(f"  Read: {read_bytes / 1024**2:.2f} MB")
        print(f"  Write: {write_bytes / 1024**2:.2f} MB")
        
        if disk_read_samples and disk_write_samples:
            read_rates = [b2 - b1 for b1, b2 in zip(disk_read_samples, disk_read_samples[1:])] if len(disk_read_samples) > 1 else [0]
            write_rates = [b2 - b1 for b1, b2 in zip(disk_write_samples, disk_write_samples[1:])] if len(disk_write_samples) > 1 else [0]
            
            if any(read_rates):
                print(f"  Peak read rate: {max(read_rates) / 1024**2:.2f} MB/s")
            if any(write_rates):
                print(f"  Peak write rate: {max(write_rates) / 1024**2:.2f} MB/s")
    
    # System-wide disk usage
    disk_usage = psutil.disk_usage(os.path.abspath('.'))
    print(f"Disk usage (current directory):")
    print(f"  Total: {disk_usage.total / 1024**3:.2f} GB")
    print(f"  Used: {disk_usage.used / 1024**3:.2f} GB ({disk_usage.percent}%)")
    print(f"  Free: {disk_usage.free / 1024**3:.2f} GB")
        
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

import time
import pynvml
from datetime import datetime

# Initialize NVML
pynvml.nvmlInit()

# Log file for GPU memory usage
log_file = "gpu_log.txt"

def get_gpu_memory_usage():
    # Get the handle for the first GPU (GPU 0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you're using GPU 0, change index for other GPUs
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    # Return the memory usage in MiB
    return info.used / 1024 ** 2  # Convert bytes to MiB

def log_peak_gpu_memory():
    peak_memory = 0
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"Monitoring started at {start_time}\n")

        try:
            while True:
                # Get current memory usage
                current_memory = get_gpu_memory_usage()
                peak_memory = max(peak_memory, current_memory)

                # Log the current memory usage
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - Current GPU Memory Usage: {current_memory:.2f} MiB\n")

                # Sleep for 1 second before checking again
                time.sleep(1)

        except KeyboardInterrupt:
            # When stopped, log the peak memory usage
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Monitoring stopped at {end_time}\n")
            f.write(f"Peak GPU Memory Usage: {peak_memory:.2f} MiB\n")

if __name__ == "__main__":
    log_peak_gpu_memory()

# Shutdown NVML
pynvml.nvmlShutdown()

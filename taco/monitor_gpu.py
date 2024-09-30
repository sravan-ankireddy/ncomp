import GPUtil
import time
import signal
import sys

# Global variables to store maximum memory usage and file path
max_memory_usage = 0
output_file = "gpu_memory_usage.txt"

# Function to gracefully terminate the program
def signal_handler(sig, frame):
    global max_memory_usage
    print("\nTerminating program...")
    print(f"Maximum GPU memory usage during the session: {max_memory_usage:.2f} MB")
    sys.exit(0)

# Register the signal handler for graceful termination (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Function to get GPU memory usage
def get_gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    if len(gpus) > 0:
        # Assume we are monitoring the first GPU (index 0)
        gpu_memory = gpus[0].memoryUsed  # Memory usage in MB
        return gpu_memory
    return 0

# Open the text file to log GPU memory usage
with open(output_file, 'w') as f:
    f.write("Time, GPU Memory Usage (MB)\n")

# Monitor GPU memory usage continuously
try:
    while True:
        # Get the current GPU memory usage
        memory_usage = get_gpu_memory_usage()
        # Update the maximum memory usage
        if memory_usage > max_memory_usage:
            max_memory_usage = memory_usage

        # Write the usage to the text file with a timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(output_file, 'a') as f:
            f.write(f"{timestamp}, {memory_usage:.2f} MB\n")

        # Print to console (optional)
        print(f"{timestamp} - GPU Memory Usage: {memory_usage:.2f} MB")

        # Wait for 1 second before the next measurement
        time.sleep(1)

except KeyboardInterrupt:
    # Print the maximum memory usage when the program is manually terminated
    print(f"Maximum GPU memory usage during the session: {max_memory_usage:.2f} MB")

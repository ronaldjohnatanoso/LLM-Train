#!/usr/bin/env python3
"""
External monitoring script for long-running training jobs.
Run this in a separate terminal while your training is running.

Usage: python monitor_training.py --pid <training_process_id> [--interval 60]
"""

import argparse
import time
import os
import psutil
import subprocess
import sys
from datetime import datetime

def get_gpu_stats():
    """Get GPU stats using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                 '--format=csv,noheader'], 
                                 capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n')
    except (subprocess.SubprocessError, FileNotFoundError):
        return ["GPU stats not available"]

def log_process_status(pid, log_file=None):
    """Log CPU, memory and GPU stats of a process"""
    try:
        process = psutil.Process(pid)
        
        # Process info
        process_status = process.status()
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # System info
        system_cpu = psutil.cpu_percent()
        system_memory = psutil.virtual_memory()
        system_memory_used_percent = system_memory.percent
        
        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log header
        output = [
            f"\n===== TRAINING MONITOR - {timestamp} =====",
            f"Process ID: {pid}, Status: {process_status}",
            f"Process CPU: {cpu_percent:.1f}%, System CPU: {system_cpu:.1f}%",
            f"Process Memory: {memory_mb:.1f} MB, System Memory Used: {system_memory_used_percent:.1f}%",
        ]
        
        # Get GPU stats
        gpu_stats = get_gpu_stats()
        output.append("\nGPU Stats:")
        for i, gpu in enumerate(gpu_stats):
            output.append(f"  {gpu}")
            
        # Add process threads info
        output.append("\nProcess Threads:")
        for i, thread in enumerate(process.threads()[:5]):  # Limit to first 5 threads
            output.append(f"  Thread {thread.id}: CPU Time {thread.user_time:.1f}s")
        
        # If there are more threads, show count only
        if len(process.threads()) > 5:
            output.append(f"  ... and {len(process.threads()) - 5} more threads")
            
        # Check if process is consuming resources
        if cpu_percent < 1.0 and process_status == 'running':
            output.append("\n⚠️ WARNING: Process appears to be running but CPU usage is very low!")
            
        output.append("=" * 40)
        output_str = '\n'.join(output)
        
        # Print to console
        print(output_str)
        
        # Write to log file if specified
        if log_file:
            with open(log_file, 'a') as f:
                f.write(output_str + '\n')
                
        return True
            
    except psutil.NoSuchProcess:
        print(f"Process {pid} no longer exists.")
        return False
    except Exception as e:
        print(f"Error monitoring process: {e}")
        return True  # Continue monitoring despite error

def main():
    parser = argparse.ArgumentParser(description='Monitor a training process')
    parser.add_argument('--pid', type=int, required=True, help='Process ID to monitor')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--log', type=str, default=None, help='Log file path')
    
    args = parser.parse_args()
    
    print(f"Starting monitoring of process {args.pid} every {args.interval} seconds")
    if args.log:
        print(f"Logging to: {args.log}")
        
    try:
        while True:
            if not log_process_status(args.pid, args.log):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    
if __name__ == "__main__":
    main()

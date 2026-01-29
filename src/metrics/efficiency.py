# src/metrics/efficiency.py
"""
Compute efficiency metrics for unlearning methods.
"""
import torch
from typing import Dict, Optional
import psutil, os, time

def efficiency_metrics(model, start_time: float, end_time: float, device: Optional[str] = None) -> Dict[str, float]:
    """Compute efficiency info: params, time, RAM, and GPU metrics if available."""
    params = sum(p.numel() for p in model.parameters())
    elapsed = end_time - start_time
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    
    metrics = {
        "params": int(params),
        "elapsed_sec": float(elapsed),
        "ram_mb": float(memory_mb),
    }
    
    # Add GPU metrics if CUDA is available
    if torch.cuda.is_available():
        try:
            # GPU memory metrics (in MB)
            gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            gpu_max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            metrics["gpu_allocated_mb"] = float(gpu_allocated_mb)
            metrics["gpu_reserved_mb"] = float(gpu_reserved_mb)
            metrics["gpu_max_allocated_mb"] = float(gpu_max_allocated_mb)
            
            # Try to get GPU utilization using pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assume GPU 0
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                metrics["gpu_utilization_percent"] = float(util.gpu)
                metrics["gpu_memory_utilization_percent"] = float(util.memory)
                metrics["gpu_total_memory_mb"] = float(mem_info.total / (1024 ** 2))
                metrics["gpu_used_memory_mb"] = float(mem_info.used / (1024 ** 2))
                
                pynvml.nvmlShutdown()
            except ImportError:
                # pynvml not available, skip these metrics
                pass
            except Exception as e:
                # Other pynvml errors, skip silently
                pass
                
        except Exception as e:
            # If any GPU metric collection fails, continue without them
            pass
    
    return metrics

def count_flops(model, input_shape=(1,3,32,32)):
    """Optional FLOPs counter if torchprofile or fvcore available."""
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(*input_shape)
        flops = FlopCountAnalysis(model, x).total()
        return flops
    except Exception:
        return None

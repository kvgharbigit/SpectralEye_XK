#!/usr/bin/env python3
"""
Comprehensive Training Bottleneck Diagnostic Tool

This script analyzes various aspects of your training pipeline to identify bottlenecks:
- GPU compute utilization and memory bandwidth
- Data loading and I/O performance
- CPU/GPU synchronization points
- Memory transfer speeds
- Model computation efficiency
"""

import os
# Fix OpenMP error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import time
import psutil
import pynvml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
import threading
import queue
from contextlib import contextmanager
import hydra
from omegaconf import DictConfig
import h5py
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


class GPUMonitor:
    """Monitor GPU metrics during operations"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> List[Dict]:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        metrics = []
        while not self.metrics_queue.empty():
            metrics.append(self.metrics_queue.get())
        return metrics
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to Watts
                
                # Clock speeds
                sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                
                metric = {
                    'timestamp': time.time(),
                    'gpu_util': util.gpu,
                    'memory_util': util.memory,
                    'memory_used_gb': mem_info.used / 1e9,
                    'memory_total_gb': mem_info.total / 1e9,
                    'temperature': temp,
                    'power_w': power,
                    'sm_clock_mhz': sm_clock,
                    'mem_clock_mhz': mem_clock
                }
                
                self.metrics_queue.put(metric)
                
            except Exception as e:
                print(f"GPU monitoring error: {e}")
                
            time.sleep(0.1)  # 100ms sampling rate


class BottleneckDiagnostic:
    """Main diagnostic class"""
    
    def __init__(self, device: str = 'cuda:0', output_dir: str = 'diagnostic_results'):
        self.device = torch.device(device)
        self.gpu_id = int(device.split(':')[1]) if ':' in device else 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.gpu_monitor = GPUMonitor(self.gpu_id)
        self.results = {}
        
    @contextmanager
    def profile_section(self, name: str):
        """Context manager for profiling code sections"""
        torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        self.gpu_monitor.start_monitoring()
        
        yield
        
        torch.cuda.synchronize(self.device)
        end_time = time.perf_counter()
        gpu_metrics = self.gpu_monitor.stop_monitoring()
        
        duration = end_time - start_time
        
        # Analyze GPU metrics
        if gpu_metrics:
            avg_gpu_util = np.mean([m['gpu_util'] for m in gpu_metrics])
            avg_mem_util = np.mean([m['memory_util'] for m in gpu_metrics])
            max_mem_used = max([m['memory_used_gb'] for m in gpu_metrics])
            avg_power = np.mean([m['power_w'] for m in gpu_metrics])
        else:
            avg_gpu_util = avg_mem_util = max_mem_used = avg_power = 0
            
        self.results[name] = {
            'duration': duration,
            'avg_gpu_util': avg_gpu_util,
            'avg_mem_util': avg_mem_util,
            'max_mem_used_gb': max_mem_used,
            'avg_power_w': avg_power,
            'gpu_metrics': gpu_metrics
        }
        
    def test_gpu_compute(self, sizes: List[int] = None) -> Dict:
        """Test raw GPU compute performance"""
        if sizes is None:
            sizes = [512, 1024, 2048, 4096, 8192]
            
        print("\n=== Testing GPU Compute Performance ===")
        compute_results = {}
        
        for size in sizes:
            print(f"\nMatrix multiplication: {size}x{size}")
            
            # Create random matrices
            a = torch.randn(size, size, device=self.device, dtype=torch.float32)
            b = torch.randn(size, size, device=self.device, dtype=torch.float32)
            
            # Warmup
            for _ in range(5):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            with self.profile_section(f'matmul_{size}'):
                iterations = 100 if size <= 2048 else 20
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                    
            # Calculate TFLOPS
            flops = 2 * size ** 3 * iterations  # 2n³ for matrix multiplication
            tflops = flops / (self.results[f'matmul_{size}']['duration'] * 1e12)
            
            compute_results[size] = {
                'tflops': tflops,
                'time_per_op': self.results[f'matmul_{size}']['duration'] / iterations,
                'gpu_util': self.results[f'matmul_{size}']['avg_gpu_util']
            }
            
            print(f"  TFLOPS: {tflops:.2f}")
            print(f"  GPU Utilization: {self.results[f'matmul_{size}']['avg_gpu_util']:.1f}%")
            
        return compute_results
        
    def test_memory_bandwidth(self) -> Dict:
        """Test GPU memory bandwidth"""
        print("\n=== Testing GPU Memory Bandwidth ===")
        bandwidth_results = {}
        
        sizes_gb = [0.1, 0.5, 1.0, 2.0, 4.0]
        
        for size_gb in sizes_gb:
            size_bytes = int(size_gb * 1e9)
            size_floats = size_bytes // 4
            
            print(f"\nTesting {size_gb}GB transfer")
            
            # Create tensors
            src = torch.randn(size_floats, device=self.device, dtype=torch.float32)
            dst = torch.empty_like(src)
            
            # Warmup
            for _ in range(3):
                dst.copy_(src)
            torch.cuda.synchronize()
            
            # Benchmark
            with self.profile_section(f'memcpy_{size_gb}gb'):
                iterations = 50
                for _ in range(iterations):
                    dst.copy_(src)
                    
            # Calculate bandwidth
            total_bytes = size_bytes * iterations * 2  # Read + Write
            bandwidth_gbps = total_bytes / (self.results[f'memcpy_{size_gb}gb']['duration'] * 1e9)
            
            bandwidth_results[f'{size_gb}gb'] = {
                'bandwidth_gbps': bandwidth_gbps,
                'time_per_copy': self.results[f'memcpy_{size_gb}gb']['duration'] / iterations
            }
            
            print(f"  Bandwidth: {bandwidth_gbps:.1f} GB/s")
            
        return bandwidth_results
        
    def test_host_to_device_transfer(self) -> Dict:
        """Test CPU-GPU transfer speeds"""
        print("\n=== Testing Host-to-Device Transfer ===")
        transfer_results = {}
        
        sizes_mb = [1, 10, 100, 500]
        
        for size_mb in sizes_mb:
            size_bytes = int(size_mb * 1e6)
            size_floats = size_bytes // 4
            
            print(f"\nTesting {size_mb}MB transfer")
            
            # Create CPU tensor
            cpu_tensor = torch.randn(size_floats, dtype=torch.float32, pin_memory=True)
            
            # H2D transfer
            with self.profile_section(f'h2d_{size_mb}mb'):
                iterations = 100 if size_mb <= 10 else 20
                for _ in range(iterations):
                    gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
                torch.cuda.synchronize()
                
            h2d_bandwidth = (size_bytes * iterations) / (self.results[f'h2d_{size_mb}mb']['duration'] * 1e9)
            
            # D2H transfer
            with self.profile_section(f'd2h_{size_mb}mb'):
                for _ in range(iterations):
                    cpu_result = gpu_tensor.to('cpu', non_blocking=True)
                torch.cuda.synchronize()
                
            d2h_bandwidth = (size_bytes * iterations) / (self.results[f'd2h_{size_mb}mb']['duration'] * 1e9)
            
            transfer_results[f'{size_mb}mb'] = {
                'h2d_bandwidth_gbps': h2d_bandwidth,
                'd2h_bandwidth_gbps': d2h_bandwidth
            }
            
            print(f"  H2D Bandwidth: {h2d_bandwidth:.2f} GB/s")
            print(f"  D2H Bandwidth: {d2h_bandwidth:.2f} GB/s")
            
        return transfer_results
        
    def test_data_loading(self, data_dir: str, num_workers_list: List[int] = None) -> Dict:
        """Test data loading performance with different configurations"""
        print("\n=== Testing Data Loading Performance ===")
        
        if num_workers_list is None:
            num_workers_list = [0, 1, 2, 4, 8, 16]
            
        loading_results = {}
        
        # Create a simple dataset that mimics your hyperspectral data loading
        class BenchmarkDataset(torch.utils.data.Dataset):
            def __init__(self, data_dir: str, size: int = 1000):
                self.data_dir = Path(data_dir)
                self.size = size
                
                # Find some sample h5 files
                self.h5_files = list(self.data_dir.glob('**/*.h5'))[:10]
                if not self.h5_files:
                    # Create dummy data if no h5 files found
                    self.use_dummy = True
                else:
                    self.use_dummy = False
                    
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                if self.use_dummy:
                    # Simulate data loading
                    time.sleep(0.001)  # Simulate I/O
                    spectral = np.random.randn(240, 240, 224).astype(np.float32)
                    rgb = np.random.randn(240, 240, 3).astype(np.float32)
                else:
                    # Load from actual h5 file
                    file_idx = idx % len(self.h5_files)
                    with h5py.File(self.h5_files[file_idx], 'r') as f:
                        # Simulate loading a patch
                        if 'data' in f:
                            data = f['data'][:]
                            h, w = 240, 240
                            if data.shape[0] >= h and data.shape[1] >= w:
                                spectral = data[:h, :w, :224]
                            else:
                                spectral = np.random.randn(h, w, 224).astype(np.float32)
                        else:
                            spectral = np.random.randn(240, 240, 224).astype(np.float32)
                    rgb = np.random.randn(240, 240, 3).astype(np.float32)
                    
                return torch.from_numpy(spectral), torch.from_numpy(rgb)
                
        dataset = BenchmarkDataset(data_dir)
        
        for num_workers in num_workers_list:
            print(f"\nTesting with {num_workers} workers")
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=8,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False
            )
            
            # Warmup
            for i, _ in enumerate(dataloader):
                if i >= 5:
                    break
                    
            # Benchmark
            with self.profile_section(f'dataload_workers_{num_workers}'):
                num_batches = 50
                for i, (spectral, rgb) in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    # Transfer to GPU to simulate real training
                    spectral = spectral.to(self.device, non_blocking=True)
                    rgb = rgb.to(self.device, non_blocking=True)
                    torch.cuda.synchronize()
                    
            samples_per_sec = (num_batches * 8) / self.results[f'dataload_workers_{num_workers}']['duration']
            
            loading_results[f'{num_workers}_workers'] = {
                'samples_per_sec': samples_per_sec,
                'time_per_batch': self.results[f'dataload_workers_{num_workers}']['duration'] / num_batches
            }
            
            print(f"  Samples/sec: {samples_per_sec:.1f}")
            print(f"  Time per batch: {loading_results[f'{num_workers}_workers']['time_per_batch']*1000:.1f}ms")
            
        return loading_results
        
    def test_model_forward_pass(self, model_sizes: Dict[str, Tuple[int, int]] = None) -> Dict:
        """Test model forward pass performance"""
        print("\n=== Testing Model Forward Pass ===")
        
        if model_sizes is None:
            # Common vision transformer configurations
            model_sizes = {
                'tiny': (192, 12),    # embed_dim, depth
                'small': (384, 12),
                'base': (768, 12),
                'large': (1024, 24)
            }
            
        forward_results = {}
        
        # Simple ViT-like model for benchmarking
        class BenchmarkViT(nn.Module):
            def __init__(self, embed_dim: int, depth: int, num_heads: int = 8, patch_size: int = 16):
                super().__init__()
                self.patch_embed = nn.Conv2d(224, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, batch_first=True)
                    for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim)
                
            def forward(self, x):
                # x: [B, C, H, W]
                x = self.patch_embed(x)  # [B, embed_dim, H', W']
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
                
                for block in self.blocks:
                    x = block(x)
                    
                x = self.norm(x)
                return x
                
        for name, (embed_dim, depth) in model_sizes.items():
            print(f"\nTesting {name} model (embed_dim={embed_dim}, depth={depth})")
            
            model = BenchmarkViT(embed_dim, depth).to(self.device)
            model.eval()
            
            # Input tensor (spectral data)
            batch_size = 8
            x = torch.randn(batch_size, 224, 240, 240, device=self.device)
            
            # Reshape for model (assuming we use first 240x240 spatial and all spectral)
            x_reshaped = x.permute(0, 1, 2, 3)[:, :, :240, :240]  # [B, C, H, W]
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x_reshaped)
                    
            # Benchmark forward pass
            with torch.no_grad():
                with self.profile_section(f'forward_{name}'):
                    iterations = 50
                    for _ in range(iterations):
                        output = model(x_reshaped)
                        torch.cuda.synchronize()
                        
            # Benchmark forward + backward
            x_reshaped.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters())
            
            with self.profile_section(f'forward_backward_{name}'):
                iterations = 20
                for _ in range(iterations):
                    output = model(x_reshaped)
                    loss = output.mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    
            forward_time = self.results[f'forward_{name}']['duration'] / 50
            forward_backward_time = self.results[f'forward_backward_{name}']['duration'] / 20
            
            forward_results[name] = {
                'forward_time_ms': forward_time * 1000,
                'forward_backward_time_ms': forward_backward_time * 1000,
                'forward_gpu_util': self.results[f'forward_{name}']['avg_gpu_util'],
                'backward_gpu_util': self.results[f'forward_backward_{name}']['avg_gpu_util']
            }
            
            print(f"  Forward pass: {forward_time*1000:.1f}ms")
            print(f"  Forward+Backward: {forward_backward_time*1000:.1f}ms")
            print(f"  GPU Utilization (forward): {forward_results[name]['forward_gpu_util']:.1f}%")
            
        return forward_results
        
    def test_mixed_precision(self) -> Dict:
        """Test mixed precision training performance"""
        print("\n=== Testing Mixed Precision Performance ===")
        mp_results = {}
        
        # Create a model
        model = nn.Sequential(
            nn.Conv2d(224, 384, 16, 16),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(384, 8, 1536, batch_first=True),
                num_layers=12
            )
        ).to(self.device)
        
        x = torch.randn(8, 224, 240, 240, device=self.device)
        
        # Test FP32
        print("\nTesting FP32 training")
        optimizer = torch.optim.AdamW(model.parameters())
        
        with self.profile_section('fp32_training'):
            iterations = 20
            for _ in range(iterations):
                output = model(x)
                loss = output.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
        fp32_time = self.results['fp32_training']['duration'] / iterations
        
        # Test FP16 with autocast
        print("\nTesting Mixed Precision (FP16) training")
        model.zero_grad()
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        with self.profile_section('fp16_training'):
            iterations = 20
            for _ in range(iterations):
                with torch.cuda.amp.autocast():
                    output = model(x)
                    loss = output.mean()
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        fp16_time = self.results['fp16_training']['duration'] / iterations
        
        mp_results = {
            'fp32_time_ms': fp32_time * 1000,
            'fp16_time_ms': fp16_time * 1000,
            'speedup': fp32_time / fp16_time,
            'fp32_memory_gb': self.results['fp32_training']['max_mem_used_gb'],
            'fp16_memory_gb': self.results['fp16_training']['max_mem_used_gb']
        }
        
        print(f"\nMixed Precision Speedup: {mp_results['speedup']:.2f}x")
        print(f"Memory Saved: {mp_results['fp32_memory_gb'] - mp_results['fp16_memory_gb']:.1f}GB")
        
        return mp_results
        
    def analyze_and_report(self):
        """Analyze results and generate report"""
        print("\n=== Generating Diagnostic Report ===")
        
        report = {
            'timestamp': self.timestamp,
            'device': str(self.device),
            'gpu_name': torch.cuda.get_device_name(self.device),
            'results': self.results,
            'analysis': {}
        }
        
        # Analyze bottlenecks
        bottlenecks = []
        
        # Check GPU utilization
        avg_gpu_utils = [r['avg_gpu_util'] for r in self.results.values() if 'avg_gpu_util' in r]
        if avg_gpu_utils:
            overall_gpu_util = np.mean(avg_gpu_utils)
            if overall_gpu_util < 80:
                bottlenecks.append(f"Low GPU utilization ({overall_gpu_util:.1f}%). Consider:")
                bottlenecks.append("  - Increasing batch size")
                bottlenecks.append("  - Reducing data loading overhead")
                bottlenecks.append("  - Using mixed precision training")
                
        # Check memory bandwidth utilization
        if 'memcpy_2.0gb' in self.results:
            bandwidth = self.results['memcpy_2.0gb'].get('bandwidth_gbps', 0)
            # Most modern GPUs have >500GB/s bandwidth
            if bandwidth < 400:
                bottlenecks.append(f"Low memory bandwidth utilization ({bandwidth:.0f}GB/s)")
                
        # Check data loading
        if any('dataload_workers' in k for k in self.results.keys()):
            # Find optimal number of workers
            worker_perfs = [(int(k.split('_')[2]), v.get('samples_per_sec', 0)) 
                           for k, v in self.results.items() if 'dataload_workers' in k]
            if worker_perfs:
                optimal_workers = max(worker_perfs, key=lambda x: x[1])[0]
                bottlenecks.append(f"Optimal data loader workers: {optimal_workers}")
                
        report['analysis']['bottlenecks'] = bottlenecks
        
        # Save JSON report
        report_path = self.output_dir / f'diagnostic_report_{self.timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Generate visualizations
        self._generate_visualizations()
        
        print(f"\nReport saved to: {report_path}")
        print("\n=== Key Findings ===")
        for bottleneck in bottlenecks:
            print(f"- {bottleneck}")
            
        return report
        
    def _generate_visualizations(self):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # GPU Utilization over time
        ax = axes[0, 0]
        for name, result in self.results.items():
            if 'gpu_metrics' in result and result['gpu_metrics']:
                metrics = result['gpu_metrics']
                times = [(m['timestamp'] - metrics[0]['timestamp']) for m in metrics]
                gpu_utils = [m['gpu_util'] for m in metrics]
                ax.plot(times, gpu_utils, label=name[:20])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization During Operations')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Memory usage
        ax = axes[0, 1]
        mem_data = [(k, v['max_mem_used_gb']) for k, v in self.results.items() 
                    if 'max_mem_used_gb' in v]
        if mem_data:
            labels, values = zip(*mem_data)
            ax.bar(range(len(labels)), values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Memory Used (GB)')
            ax.set_title('Peak Memory Usage by Operation')
            
        # Data loading performance
        ax = axes[1, 0]
        worker_data = [(int(k.split('_')[2]), v.get('samples_per_sec', 0)) 
                       for k, v in self.results.items() if 'dataload_workers' in k]
        if worker_data:
            worker_data.sort()
            workers, samples_sec = zip(*worker_data)
            ax.plot(workers, samples_sec, 'o-')
            ax.set_xlabel('Number of Workers')
            ax.set_ylabel('Samples/sec')
            ax.set_title('Data Loading Performance')
            ax.grid(True)
            
        # Operation timings
        ax = axes[1, 1]
        timing_data = [(k, v['duration']) for k, v in self.results.items() 
                       if 'duration' in v and not k.startswith('dataload')][:10]
        if timing_data:
            labels, values = zip(*timing_data)
            ax.barh(range(len(labels)), values)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Duration (s)')
            ax.set_title('Operation Timings')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / f'diagnostic_plots_{self.timestamp}.png', dpi=150)
        plt.close()
        
        
def run_diagnostic(cfg: Optional[DictConfig] = None):
    """Run the complete diagnostic suite"""
    
    # Use config if provided, otherwise use defaults
    device = cfg.get('device', 'cuda:0') if cfg else 'cuda:0'
    output_dir = cfg.get('output_dir', 'diagnostic_results') if cfg else 'diagnostic_results'
    data_dir = cfg.get('data_dir', 'data') if cfg else 'data'
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This diagnostic requires a GPU.")
        return
        
    diag = BottleneckDiagnostic(device=device, output_dir=output_dir)
    
    print(f"Running diagnostic on: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run all tests
    diag.test_gpu_compute()
    diag.test_memory_bandwidth()
    diag.test_host_to_device_transfer()
    diag.test_data_loading(data_dir)
    diag.test_model_forward_pass()
    diag.test_mixed_precision()
    
    # Generate report
    report = diag.analyze_and_report()
    
    return report


@hydra.main(version_base="1.3", config_path="conf", config_name="config", )
def main(cfg: DictConfig):
    """Main entry point with Hydra config support"""
    run_diagnostic(cfg)


if __name__ == '__main__':
    # Can run standalone without Hydra
    import sys
    if len(sys.argv) == 1:
        # Run with defaults if no Hydra config
        report = run_diagnostic()
    else:
        # Run with Hydra
        main()
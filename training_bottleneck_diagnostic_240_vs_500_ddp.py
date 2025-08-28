#!/usr/bin/env python3
"""
Comprehensive 240x240 vs 500x500 Training Bottleneck Diagnostic with DDP
This script tests both spatial configurations using actual 3-GPU DDP setup.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["TORCH_DISTRIBUTED_USE_LIBUV"] = "0"
os.environ["USE_LIBUV"] = "0"
os.environ["GLOO_SOCKET_IFNAME"] = ""

import sys
from pathlib import Path

# Add parent directory to path so we can import model_training
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import time
import pynvml
from datetime import datetime, timedelta
import json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp_native
from contextlib import contextmanager
import threading
import queue
import psutil

# Set the sharing strategy to use the file system
mp.set_sharing_strategy('file_system')

# Import your actual model and dataset
from model_training.models.spectral_gpt.spectral_gpt import MaskedAutoencoderViT
from model_training.dataset.combined_dataset import get_dataset
from model_training.losses.custom_loss import CustomLoss


class DDPBottleneckDiagnostic:
    """Comprehensive diagnostic comparing 240x240 vs 500x500 configurations with actual DDP"""
    
    def __init__(self, base_cfg: DictConfig):
        self.base_cfg = base_cfg
        self.output_dir = Path('ddp_diagnostic_results')
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize GPU monitoring
        pynvml.nvmlInit()
        self.results = {}
        self.all_results = {}  # Track results across all tests
        
    def log_both(self, message: str, rank: int = 0):
        """Log message to both console and text file (only from rank 0)"""
        if rank == 0:
            print(message)
            # Write to file if output_file is set
            if hasattr(self, 'output_file'):
                with open(self.output_file, 'a') as f:
                    f.write(message + '\n')
        
    def get_gpu_metrics(self, device_id: int) -> Dict:
        """Get current GPU metrics"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            
            return {
                'gpu_util': util.gpu,
                'memory_util': util.memory,
                'memory_used_gb': mem_info.used / 1e9,
                'memory_free_gb': mem_info.free / 1e9,
                'temperature': temp,
                'power_w': power
            }
        except:
            return {}
    
    def setup_ddp(self, rank: int, world_size: int):
        """Initialize DDP process group exactly like training code"""
        from torch.distributed import TCPStore
        
        timeout = timedelta(seconds=300)
        
        if rank == 0:
            store = TCPStore("127.0.0.1", 29501,  # Different port to avoid conflict
                           world_size=world_size, is_master=True, 
                           timeout=timeout, wait_for_workers=True, use_libuv=False)
        else:
            store = TCPStore("127.0.0.1", 29501, 
                           world_size=world_size, is_master=False, 
                           timeout=timeout, wait_for_workers=True, use_libuv=False)
        
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=rank,
            world_size=world_size
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        
    def cleanup_ddp(self):
        """Clean up DDP process group"""
        if dist.is_initialized():
            dist.destroy_process_group()
            
    def test_configuration_ddp(self, rank: int, world_size: int, cfg: DictConfig, 
                             spatial_size: int, model_name: str, result_queue: mp.Queue):
        """Test a configuration using actual DDP setup"""
        try:
            # Setup DDP
            self.setup_ddp(rank, world_size)
            device = torch.device(f"cuda:{rank}")
            
            # Only rank 0 prints
            if rank == 0:
                print(f"\n=== TESTING {spatial_size}x{spatial_size} - {model_name.upper()} with {world_size} GPUs ===")
                print(f"Model config: {cfg.model.name}")
                print(f"Dataset path: {cfg.dataset.csv_path}")
                print(f"Image size: {spatial_size}")
                print(f"Spatial patch size: {cfg.model.model.spatial_patch_size}")
                print(f"Wavelength patch size: {cfg.model.model.wavelength_patch_size}")
            
            # Create dataset (each process loads full dataset)
            dataset_fn = instantiate(cfg.dataset)
            datasets = dataset_fn(transform=None)
            train_dataset = datasets[0]
            
            # Create preprocessor
            def preprocess_hsi(x):
                # Apply log transform and normalization
                x = torch.log1p(x)
                x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-8)
                return x
            
            # Create distributed sampler (ensures each GPU gets different data)
            sampler = DistributedSampler(
                train_dataset, 
                num_replicas=world_size,
                rank=rank,
                shuffle=False  # Consistent for benchmarking
            )
            
            # Create dataloader with actual training parameters
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.hparams.batch_size,  # Per-GPU batch size
                num_workers=cfg.dataloader.num_workers,
                pin_memory=cfg.dataloader.pin_memory if hasattr(cfg.dataloader, 'pin_memory') else False,
                prefetch_factor=cfg.dataloader.prefetch_factor if cfg.dataloader.num_workers > 0 else None,
                persistent_workers=cfg.dataloader.persistent_workers if cfg.dataloader.num_workers > 1 else False,
                shuffle=False,
                sampler=sampler,
                timeout=60 if cfg.dataloader.num_workers > 0 else 0
            )
            
            # Test different batch sizes
            test_batches = [1, 2, 4, 6, 8, 12, 16, 24, 32]
            max_batch = 1
            
            if rank == 0:
                print(f"\nFinding maximum batch size for {model_name} at {spatial_size}x{spatial_size}...")
            
            # Find max batch size
            for batch_size in test_batches:
                try:
                    torch.cuda.empty_cache()
                    
                    # Quick test with temporary model
                    temp_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                    temp_cfg.hparams.batch_size = batch_size
                    
                    # Create model using instantiate (handles the model.model structure)
                    model = instantiate(temp_cfg.model.model).to(device)
                    
                    # Wrap with DDP
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
                    
                    # Get one batch and test
                    for batch in dataloader:
                        hs_cube, label, rgb = batch
                        hs_cube = preprocess_hsi(hs_cube)
                        hs_cube = hs_cube.to(device, non_blocking=True)
                        
                        # Test forward pass
                        output = model(hs_cube)
                        torch.cuda.synchronize()
                        
                        # Get memory usage
                        metrics = self.get_gpu_metrics(rank)
                        memory_gb = metrics.get('memory_used_gb', 0)
                        
                        # All processes must agree on success
                        success_tensor = torch.tensor([1.0], device=device)
                        dist.all_reduce(success_tensor)
                        
                        if success_tensor.item() == world_size:
                            max_batch = batch_size
                            if rank == 0:
                                print(f"  Batch {batch_size}: OK ({memory_gb:.1f}GB)")
                        break
                        
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    # Synchronize failure across processes
                    failure_tensor = torch.tensor([0.0], device=device)
                    dist.all_reduce(failure_tensor)
                    
                    if rank == 0:
                        print(f"  Batch {batch_size}: OOM - Max batch size is {max_batch}")
                    break
            
            # Now test performance with different configurations
            if rank == 0:
                print(f"\n=== MODEL PERFORMANCE TESTING ({spatial_size}x{spatial_size}) - {model_name.upper()} - DDP {world_size} GPUs ===")
                print(f"Testing batch sizes: {test_batches[:test_batches.index(max_batch)+1]} (per-GPU)")
                print("Config               Data Load    Model FWD    Model BWD    GPU Util   Memory     Training/s   Total/s      Epoch(34k)   ")
                print("-" * 125)
            
            # Test each configuration
            worker_configs = [1, 2, 4]
            batch_configs = test_batches[:test_batches.index(max_batch)+1]
            
            best_config = None
            best_throughput = 0
            
            for num_workers in worker_configs:
                for batch_size in batch_configs:
                    config_name = f"w{num_workers}_b{batch_size}"
                    
                    if rank == 0:
                        print(f"  Testing {config_name}...", end='', flush=True)
                    
                    # Test this configuration
                    metrics = self.benchmark_configuration_ddp(
                        rank, world_size, cfg, batch_size, num_workers, 
                        spatial_size, model_name, device, preprocess_hsi
                    )
                    
                    # Gather results from all processes
                    if rank == 0:
                        all_metrics = [metrics]
                        for i in range(1, world_size):
                            recv_metrics = {}
                            dist.recv(recv_metrics, src=i)
                            all_metrics.append(recv_metrics)
                        
                        # Average metrics across GPUs
                        avg_metrics = self.average_metrics(all_metrics)
                        
                        # Display results
                        total_throughput = avg_metrics['samples_per_sec'] * world_size
                        print(f"\r{config_name:<20} {avg_metrics['data_rate']:<12.1f} "
                              f"{avg_metrics['forward_time']:<12.1f} "
                              f"{avg_metrics['backward_time']:<12.1f} "
                              f"{avg_metrics['gpu_util']:<10.0f} "
                              f"{avg_metrics['memory_gb']:<10.1f} "
                              f"{avg_metrics['samples_per_sec']:<12.1f} "
                              f"{total_throughput:<12.1f} "
                              f"{avg_metrics['epoch_time']:<14}")
                        
                        # Track best configuration
                        if total_throughput > best_throughput:
                            best_throughput = total_throughput
                            best_config = {
                                'workers': num_workers,
                                'batch_size': batch_size,
                                'total_throughput': total_throughput,
                                'config': config_name,
                                'data_rate': avg_metrics['data_rate'],
                                'gpu_util': avg_metrics['gpu_util']
                            }
                    else:
                        # Send metrics to rank 0
                        dist.send(metrics, dst=0)
                    
                    # Synchronize before next test
                    dist.barrier()
            
            # Report best configuration
            if rank == 0 and best_config:
                print(f"\nBest configuration: {best_config['config']} "
                      f"({best_config['total_throughput']:.1f} samples/s)")
                
                # Results are displayed inline above
            
            # Clean up
            self.cleanup_ddp()
            
            # Send completion signal
            if rank == 0:
                result_queue.put("DONE")
                
        except Exception as e:
            print(f"Error in rank {rank}: {str(e)}")
            if dist.is_initialized():
                self.cleanup_ddp()
            if rank == 0:
                result_queue.put(f"ERROR: {str(e)}")
    
    def get_disk_io_stats(self):
        """Get disk I/O statistics for F: drive"""
        try:
            # Get disk I/O counters
            disk_io = psutil.disk_io_counters(perdisk=True)
            
            # Find F: drive stats (Windows)
            for disk_name, stats in disk_io.items():
                # On Windows, physical drives are named like 'PhysicalDrive0'
                # We'll sum all disk I/O as F: could be on any physical disk
                return {
                    'read_mb_per_sec': stats.read_bytes / 1024 / 1024,
                    'write_mb_per_sec': stats.write_bytes / 1024 / 1024,
                    'read_count': stats.read_count,
                    'write_count': stats.write_count,
                    'busy_time': getattr(stats, 'busy_time', 0)
                }
            return {}
        except:
            return {}
    
    def benchmark_configuration_ddp(self, rank: int, world_size: int, cfg: DictConfig,
                                   batch_size: int, num_workers: int, spatial_size: int,
                                   model_name: str, device: torch.device, 
                                   preprocess_hsi) -> Dict:
        """Benchmark a specific configuration with DDP"""
        try:
            # Update config
            test_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            
            # Ensure we're using full dataset
            test_cfg.dataset.trial_mode = False
            
            # Update test parameters
            test_cfg.hparams.batch_size = batch_size
            test_cfg.dataloader.num_workers = num_workers
            
            # Create dataset
            dataset_fn = instantiate(test_cfg.dataset)
            datasets = dataset_fn(transform=None)
            train_dataset = datasets[0]
            
            # Create distributed sampler
            sampler = DistributedSampler(
                train_dataset, 
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=num_workers > 1,
                shuffle=False,
                sampler=sampler,
                timeout=60 if num_workers > 0 else 0
            )
            
            # Create model using instantiate (handles the model.model structure)
            model = instantiate(test_cfg.model.model).to(device)
            
            # Wrap with DDP
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            
            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=test_cfg.optimizer.lr)
            
            # Measure data loading rate
            data_times = []
            forward_times = []
            backward_times = []
            gpu_utils = []
            
            # Warmup
            for i, batch in enumerate(dataloader):
                if i >= 3:
                    break
                hs_cube, label, rgb = batch
                hs_cube = preprocess_hsi(hs_cube)
                hs_cube = hs_cube.to(device, non_blocking=True)
                output = model(hs_cube)
                if isinstance(output, tuple):
                    loss = output[0]
                else:
                    loss = output.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Actual benchmark
            torch.cuda.synchronize()
            for i, batch in enumerate(dataloader):
                if i >= 10:  # Test 10 iterations
                    break
                
                # Measure data loading
                data_start = time.perf_counter()
                hs_cube, label, rgb = batch
                hs_cube = preprocess_hsi(hs_cube)
                hs_cube = hs_cube.to(device, non_blocking=True)
                torch.cuda.synchronize()
                data_times.append(time.perf_counter() - data_start)
                
                # Measure forward pass
                forward_start = time.perf_counter()
                output = model(hs_cube)
                if isinstance(output, tuple):
                    loss = output[0]
                else:
                    loss = output.mean()
                torch.cuda.synchronize()
                forward_times.append(time.perf_counter() - forward_start)
                
                # Measure backward pass
                backward_start = time.perf_counter()
                loss.backward()
                
                # DDP synchronizes gradients here
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                backward_times.append(time.perf_counter() - backward_start)
                
                # Get GPU utilization
                gpu_metrics = self.get_gpu_metrics(rank)
                gpu_utils.append(gpu_metrics.get('gpu_util', 0))
            
            # Calculate metrics
            avg_data_time = np.mean(data_times)
            avg_forward_time = np.mean(forward_times) * 1000  # Convert to ms
            avg_backward_time = np.mean(backward_times) * 1000
            avg_gpu_util = np.mean(gpu_utils)
            
            # Data rate (samples per second)
            data_rate = batch_size / avg_data_time
            
            # Training throughput (considering all 3 GPUs)
            total_time_per_batch = (avg_forward_time + avg_backward_time) / 1000  # Back to seconds
            samples_per_sec = batch_size / total_time_per_batch
            total_samples_per_sec = samples_per_sec * world_size  # All GPUs combined
            
            # Epoch time estimation
            total_samples = 34000  # Based on your output
            epoch_seconds = total_samples / total_samples_per_sec
            epoch_time = f"{int(epoch_seconds/60)}min" if epoch_seconds < 3600 else f"{epoch_seconds/3600:.1f}hr"
            
            # Get final memory usage
            final_metrics = self.get_gpu_metrics(rank)
            memory_gb = final_metrics.get('memory_used_gb', 0)
            
            return {
                'data_rate': data_rate,
                'forward_time': avg_forward_time,
                'backward_time': avg_backward_time,
                'gpu_util': avg_gpu_util,
                'memory_gb': memory_gb,
                'samples_per_sec': samples_per_sec,
                'epoch_time': epoch_time
            }
            
        except Exception as e:
            return {
                'data_rate': 0,
                'forward_time': 0,
                'backward_time': 0,
                'gpu_util': 0,
                'memory_gb': 0,
                'samples_per_sec': 0,
                'epoch_time': 'ERROR'
            }
    
    def average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average metrics across all GPUs"""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key == 'epoch_time':
                avg_metrics[key] = metrics_list[0][key]  # Use from rank 0
            else:
                values = [m[key] for m in metrics_list]
                avg_metrics[key] = np.mean(values)
        return avg_metrics
    
    def run_diagnostic(self):
        """Run the complete diagnostic with DDP"""
        # Create output file path (but don't open yet)
        self.output_file = self.output_dir / f'ddp_diagnostic_{self.timestamp}.txt'
        
        self.log_both("=== COMPREHENSIVE 240x240 vs 500x500 DDP COMPARISON ===")
        self.log_both(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_both(f"Data path: {self.base_cfg.dataset.csv_path}")
        self.log_both(f"Base configuration: {self.base_cfg.hparams.batch_size} batch, "
                     f"{self.base_cfg.dataloader.num_workers} workers")
        self.log_both(f"Available GPUs: {torch.cuda.device_count()}\n")
        
        # Test configurations
        base_path = Path(self.base_cfg.dataset.csv_path).parent.parent
        configs_to_test = [
            # (spatial_size, model_name, dataset_path)
            (240, 'mae_small_240', str(base_path / 'data_240' / 'data_all.csv')),
            (240, 'mae_medium_240', str(base_path / 'data_240' / 'data_all.csv')),
            (500, 'mae_small', str(base_path / 'data_500' / 'data_all.csv')),
            (500, 'mae_medium', str(base_path / 'data_500' / 'data_all.csv')),
        ]
        
        # Test with different numbers of GPUs
        max_gpus = min(torch.cuda.device_count(), 3)  # Test up to 3 GPUs
        gpu_counts = list(range(1, max_gpus + 1))
        
        for num_gpus in gpu_counts:
            self.log_both(f"\n{'='*80}")
            self.log_both(f"TESTING WITH {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
            self.log_both(f"{'='*80}\n")
            
            # Track best configurations for this GPU count
            best_configs_by_gpu = {}
            
            for spatial_size, model_name, dataset_path in configs_to_test:
                # Update config for this test
                test_cfg = OmegaConf.create(OmegaConf.to_container(self.base_cfg, resolve=True))
                test_cfg.model.name = model_name
                test_cfg.dataset.csv_path = dataset_path
                
                # CRITICAL: Disable trial mode for realistic performance testing
                test_cfg.dataset.trial_mode = False
                test_cfg.dataset.trial_size = 1000  # Not used when trial_mode=False
                
                # Ensure proper dataloader settings (keep existing structure)
                if not hasattr(test_cfg.dataloader, 'pin_memory'):
                    test_cfg.dataloader.pin_memory = False  # Disabled for DDP
                if not hasattr(test_cfg.dataloader, 'prefetch_factor'):
                    test_cfg.dataloader.prefetch_factor = 4
                if not hasattr(test_cfg.dataloader, 'persistent_workers'):
                    test_cfg.dataloader.persistent_workers = True
                
                # Adjust patch sizes based on spatial size
                if spatial_size == 240:
                    test_cfg.model.model.spatial_patch_size = 12
                    test_cfg.model.model.img_size = 240
                else:
                    test_cfg.model.model.spatial_patch_size = 25
                    test_cfg.model.model.img_size = 500
                
                # Use multiprocessing queue for results
                result_queue = mp.Queue()
                
                # Spawn processes
                mp.spawn(
                    self.test_configuration_ddp,
                    args=(num_gpus, test_cfg, spatial_size, model_name, result_queue),
                    nprocs=num_gpus,
                    join=True
                )
                
                # Get result
                try:
                    result = result_queue.get(timeout=300)
                    if result != "DONE":
                        self.log_both(f"Error testing {model_name}: {result}")
                    else:
                        # Just log that the test completed successfully
                        key = f"{model_name}_{spatial_size}"
                        self.log_both(f"✓ Completed {key} with {num_gpus} GPU(s)")
                except:
                    self.log_both(f"Timeout testing {model_name} with {num_gpus} GPUs")
                
                # Add some spacing between tests
                self.log_both("")
            
            # Best configurations are shown inline in each test above
            self.log_both(f"\n=== {num_gpus} GPU TESTING COMPLETED ===")
            self.log_both("Check individual test results above for best configurations per model.")
        
        # Summary analysis
        self.log_both(f"\n{'='*80}")
        self.log_both("SUMMARY ANALYSIS")
        self.log_both(f"{'='*80}")
        
        # Print summary guidance
        self.log_both("\n=== OPTIMIZATION GUIDANCE ===")
        self.log_both("\nTo analyze your results, look for:")
        self.log_both("1. Best configurations printed after each model test")
        self.log_both("2. Total throughput (Total/s) scaling patterns:")
        self.log_both("   - 1 GPU → 2 GPUs: Should roughly double")
        self.log_both("   - 2 GPUs → 3 GPUs: Should increase by ~50%")
        self.log_both("   - If throughput plateaus = I/O bottleneck confirmed")
        self.log_both("3. Data Load rates: Should stay consistent across GPU counts")
        self.log_both("4. GPU Utilization: Should stay >80% if no I/O starvation")
        
        self.log_both("\nKey findings:")
        self.log_both("1. Data Load Rate: Check if it decreases with more GPUs (F: drive bottleneck)")
        self.log_both("2. GPU Utilization: Should stay >80% if well-fed with data")
        self.log_both("3. Scaling Efficiency:")
        self.log_both("   - Linear scaling (2x, 3x) = No I/O bottleneck")
        self.log_both("   - Sub-linear scaling = I/O or communication bottleneck")
        self.log_both("4. Optimal Worker Count:")
        self.log_both("   - Usually 1-2 workers per GPU is optimal with shared storage")
        self.log_both("   - More workers may increase disk contention")
        
        self.log_both(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults saved to: {self.output_file}")


@hydra.main(version_base="1.3", config_path="model_training/conf", config_name="full_run_240")
def main(cfg: DictConfig) -> None:
    """Run the comprehensive DDP diagnostic"""
    # Force DDP mode and proper device configuration
    cfg.general.use_ddp = True
    cfg.general.parallel.use_parallel = True
    cfg.general.parallel.device_ids = [0, 1, 2]  # Use all 3 GPUs
    
    # Disable trial mode for realistic benchmarking
    cfg.dataset.trial_mode = False
    
    # Print configuration being used
    print(f"Using configuration:")
    print(f"- Model: {cfg.model.name}")
    print(f"- Dataset: {cfg.dataset.csv_path}")
    print(f"- DDP enabled: {cfg.general.use_ddp}")
    print(f"- Trial mode: {cfg.dataset.trial_mode}")
    print(f"- Base batch size: {cfg.hparams.batch_size}")
    print(f"- Base workers: {cfg.dataloader.num_workers}")
    print("-" * 50)
    
    # Create diagnostic instance
    diagnostic = DDPBottleneckDiagnostic(cfg)
    
    # Run diagnostic
    diagnostic.run_diagnostic()


if __name__ == "__main__":
    main()
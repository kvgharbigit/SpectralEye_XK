#!/usr/bin/env python3
"""
Comprehensive 240x240 vs 500x500 Training Bottleneck Diagnostic

This script tests both spatial configurations to find optimal settings including model performance.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# Add parent directory to path so we can import model_training
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import torch
import torch.nn as nn
import numpy as np
import time
import pynvml
from datetime import datetime
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional
import multiprocessing as mp
from contextlib import contextmanager

# Import your actual model and dataset
from model_training.models.spectral_gpt.spectral_gpt import MaskedAutoencoderViT
from model_training.dataset.combined_dataset import get_dataset
from model_training.losses.custom_loss import CustomLoss


class ComprehensiveBottleneckDiagnostic:
    """Comprehensive diagnostic comparing 240x240 vs 500x500 configurations"""
    
    def __init__(self, base_cfg: DictConfig):
        self.base_cfg = base_cfg
        self.output_dir = Path('comprehensive_diagnostic_results')  # Results in same directory as script
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize GPU monitoring
        pynvml.nvmlInit()
        self.results = {}
        self.text_file = None
        
    def log_both(self, message: str):
        """Log message to both console and text file"""
        print(message)
        if self.text_file:
            self.text_file.write(message + '\n')
            self.text_file.flush()
        
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
            
    def test_configuration_matrix(self, spatial_size: int) -> Dict:
        """Test comprehensive matrix of configurations for given spatial size"""
        self.log_both(f"\n=== TESTING {spatial_size}x{spatial_size} CONFIGURATION ===")
        
        # Create configuration for this spatial size
        from copy import deepcopy
        cfg = OmegaConf.create(OmegaConf.to_yaml(self.base_cfg))
        cfg.spatial_size = spatial_size
        
        # Update model config based on spatial size
        if spatial_size == 240:
            # Load 240x240 model config
            model_config = OmegaConf.load('model_training/conf/model/mae_medium_240.yaml')
        else:
            # Use default 500x500 config
            model_config = OmegaConf.load('model_training/conf/model/mae_medium.yaml')
        
        # Override model config
        cfg.model = model_config
        
        self.log_both(f"Model config: {cfg.model.name}")
        self.log_both(f"Image size: {cfg.model.model.img_size}")
        self.log_both(f"Spatial patch size: {cfg.model.model.spatial_patch_size}")
        self.log_both(f"Wavelength patches: {cfg.model.model.num_wavelengths}")
        
        # Test different configurations
        worker_configs = [1, 2, 4]  # Remove 8 workers - diminishing returns
        batch_sizes = [1, 2, 4, 6]  # Test batch sizes up to 6
        
        results = {}
        best_data_config = None
        best_data_throughput = 0
        
        # MOVED: Test model performance first for faster debugging
        self.log_both(f"\n=== MODEL PERFORMANCE TESTING ({spatial_size}x{spatial_size}) ===")
        self.log_both(f"{'Config':<20} {'Data Rate':<12} {'Model FWD':<12} {'Model BWD':<12} {'GPU Util':<10} {'Memory':<10} {'Overall':<12}")
        self.log_both("-" * 100)
        
        # Test a few representative configs for model performance
        test_configs = [
            (2, 6),  # 2 workers, batch 6
            (4, 6),  # 4 workers, batch 6  
            (1, 6),  # 1 worker, batch 6
        ]
        
        for num_workers, batch_size in test_configs:
            config_key = f"{spatial_size}x{spatial_size}_w{num_workers}_b{batch_size}"
            self.log_both(f"  Testing model performance for {config_key}...")
            model_results = self.test_model_performance(cfg, batch_size, num_workers)
            
            if 'error' not in model_results:
                data_rate = 0  # Will be filled by data loading test later
                fwd_time = model_results.get('forward_time_ms', 0)
                bwd_time = model_results.get('training_step_time_ms', 0) - fwd_time
                gpu_util = model_results.get('gpu_util', 0)
                memory = model_results.get('memory_gb', 0)
                
                if model_results.get('training_step_time_ms', 0) > 0:
                    training_samples_per_sec = (batch_size * 1000) / model_results['training_step_time_ms']
                    overall = f"{training_samples_per_sec:.0f}/s"
                else:
                    overall = "ERROR"
                
                self.log_both(f"{config_key:<20} {'TBD':<12} {fwd_time:<12.1f} {bwd_time:<12.1f} {gpu_util:<10.0f} {memory:<10.1f} {overall:<12}")
            else:
                self.log_both(f"{config_key:<20} {'TBD':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
        
        self.log_both(f"\n=== DATA LOADING PERFORMANCE ({spatial_size}x{spatial_size}) ===")
        self.log_both(f"{'Workers':<8} {'Batch':<6} {'Samples/s':<12} {'Batch Time':<12} {'I/O %':<8} {'Status':<15}")
        self.log_both("-" * 75)
        
        # Create dataset once
        train_dataset, val_dataset = get_dataset(
            csv_path=cfg.dataset.csv_path,
            train_ratio=cfg.dataset.train_ratio,
            seed=cfg.dataset.seed,
            trial_mode=True,
            trial_size=200,
            transform=None
        )
        
        # Test data loading configurations
        for batch_size in batch_sizes:
            for num_workers in worker_configs:
                config_key = f"{spatial_size}x{spatial_size}_w{num_workers}_b{batch_size}"
                
                try:
                    # Memory estimation (rough)
                    channels = 224  # Assuming 224 spectral channels
                    estimated_memory = batch_size * spatial_size * spatial_size * channels * 4 / (1024**3)
                    
                    if estimated_memory > 10:  # Skip if likely to OOM
                        results[config_key] = {
                            'spatial_size': spatial_size,
                            'batch_size': batch_size,
                            'num_workers': num_workers,
                            'status': 'SKIPPED_MEMORY',
                            'samples_per_sec': 0,
                            'estimated_memory_gb': estimated_memory
                        }
                        self.log_both(f"{num_workers:<8} {batch_size:<6} {'SKIPPED':<12} {'OOM Risk':<12} {'':<8} {'Memory > 10GB':<15}")
                        continue
                    
                    dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                        persistent_workers=num_workers > 1,
                        shuffle=False,
                        timeout=30 if num_workers > 0 else 0
                    )
                    
                    # Quick warmup
                    for i, batch in enumerate(dataloader):
                        if i >= 2:
                            break
                    
                    # Benchmark data loading
                    # Use available GPU device
                    if torch.cuda.device_count() > 1:
                        device = torch.device('cuda:1')
                    else:
                        device = torch.device('cuda:0')
                    num_batches = min(15, len(dataloader))
                    batch_times = []
                    io_times = []
                    
                    start_time = time.perf_counter()
                    for i, batch in enumerate(dataloader):
                        if i >= num_batches:
                            break
                            
                        batch_start = time.perf_counter()
                        
                        # Get data
                        if isinstance(batch, dict):
                            spectral = batch['spectral']
                        else:
                            spectral = batch[0]
                            
                        io_end = time.perf_counter()
                        io_time = io_end - batch_start
                        
                        # Transfer to GPU
                        spectral_gpu = spectral.to(device, non_blocking=True)
                        torch.cuda.synchronize()
                        
                        total_time = time.perf_counter() - batch_start
                        
                        batch_times.append(total_time)
                        io_times.append(io_time)
                    
                    # Calculate metrics
                    avg_batch_time = np.mean(batch_times)
                    avg_io_time = np.mean(io_times)
                    total_time = sum(batch_times)
                    samples_per_sec = (num_batches * batch_size) / total_time if total_time > 0 else 0
                    io_overhead_pct = (avg_io_time / avg_batch_time) * 100 if avg_batch_time > 0 else 0
                    
                    results[config_key] = {
                        'spatial_size': spatial_size,
                        'batch_size': batch_size,
                        'num_workers': num_workers,
                        'samples_per_sec': samples_per_sec,
                        'avg_batch_time_ms': avg_batch_time * 1000,
                        'io_overhead_pct': io_overhead_pct,
                        'estimated_memory_gb': estimated_memory,
                        'status': 'SUCCESS'
                    }
                    
                    if samples_per_sec > best_data_throughput:
                        best_data_throughput = samples_per_sec
                        best_data_config = config_key
                    
                    # Status
                    if io_overhead_pct > 80:
                        status = "I/O BOUND"
                    elif avg_batch_time > 0.01:  # > 10ms
                        status = "SLOW"
                    else:
                        status = "GOOD"
                    
                    self.log_both(f"{num_workers:<8} {batch_size:<6} {samples_per_sec:<12.0f} {avg_batch_time*1000:<12.1f} {io_overhead_pct:<8.1f} {status:<15}")
                    
                except Exception as e:
                    results[config_key] = {
                        'spatial_size': spatial_size,
                        'status': f'ERROR: {str(e)[:20]}',
                        'samples_per_sec': 0
                    }
                    self.log_both(f"{num_workers:<8} {batch_size:<6} {'ERROR':<12} {str(e)[:30]:<30}")
                    continue
        
        self.log_both(f"\nBest data loading config: {best_data_config} ({best_data_throughput:.0f} samples/sec)")
        
        # Model testing was moved to the beginning for faster debugging
        self.log_both(f"\n=== MODEL TESTING COMPLETED ABOVE ===")
        
        return results
        
    def test_model_performance(self, cfg: DictConfig, batch_size: int, num_workers: int) -> Dict:
        """Test actual model performance with given configuration"""
        
        device = torch.device('cuda:1')  # Your configured device
        
        try:
            # Clear GPU cache before testing
            torch.cuda.empty_cache()
            
            # Get initial memory state
            memory_reserved_start = torch.cuda.memory_reserved(device) / 1e9
            memory_allocated_start = torch.cuda.memory_allocated(device) / 1e9
            
            # Create model
            model = MaskedAutoencoderViT(
                img_size=cfg.model.model.img_size,
                num_channels=cfg.model.model.num_channels,
                num_wavelengths=cfg.model.model.num_wavelengths,
                spatial_patch_size=cfg.model.model.spatial_patch_size,
                wavelength_patch_size=cfg.model.model.wavelength_patch_size,
                encoder_embed_dim=cfg.model.model.encoder_embed_dim,
                encoder_depth=cfg.model.model.encoder_depth,
                encoder_num_heads=cfg.model.model.encoder_num_heads,
                decoder_embed_dim=cfg.model.model.decoder_embed_dim,
                decoder_depth=cfg.model.model.decoder_depth,
                decoder_num_heads=cfg.model.model.decoder_num_heads,
                mlp_ratio=cfg.model.model.mlp_ratio,
                mask_ratio=cfg.model.model.mask_ratio
            ).to(device)
            
            # Use DataParallel if configured
            if cfg.general.parallel.use_parallel:
                model = model.to('cuda:0')
                model = nn.DataParallel(model, device_ids=cfg.general.parallel.device_ids)
                device = torch.device('cuda:0')
            
            # Create input tensor with correct dimensions
            img_size = cfg.model.model.img_size
            
            # Debug: print model expectations
            print(f"DEBUG - Model expects:")
            print(f"  img_size: {img_size}")
            print(f"  num_wavelengths: {cfg.model.model.num_wavelengths}")
            print(f"  wavelength_patch_size: {cfg.model.model.wavelength_patch_size}")
            print(f"  spatial_patch_size: {cfg.model.model.spatial_patch_size}")
            
            # For spectral data: [B, H, W, Wavelengths]
            # The model expects num_wavelengths channels (patching happens inside the model)
            expected_channels = cfg.model.model.num_wavelengths
            
            print(f"  expected_channels: {expected_channels}")
            print(f"  creating tensor shape: [{batch_size}, {expected_channels}, {img_size}, {img_size}]")
            
            # Model expects: [batch_size, wavelengths, height, width] 
            # (unsqueeze(dim=1) is called inside the encoder)
            x = torch.randn(batch_size, expected_channels, img_size, img_size, device=device)
            
            model.eval()
            
            # Test forward pass
            forward_times = []
            with torch.no_grad():
                # Warmup
                for i in range(3):
                    try:
                        print(f"  Warmup {i+1}/3...")
                        output = model(x)
                    # Model returns (loss, pred, mask) tuple
                        if isinstance(output, tuple):
                            print(f"  Success! Output tuple with {len(output)} elements")
                        else:
                            print(f"  Success! Output shape: {output.shape}")
                    except Exception as e:
                        print(f"  Forward pass failed: {str(e)}")
                        return {'error': f'Forward pass failed: {str(e)}'}
                        
                # Benchmark forward
                print("  Running forward benchmark...")
                torch.cuda.synchronize()
                for _ in range(10):
                    start_time = time.perf_counter()
                    output = model(x)
                    # Model returns (loss, pred, mask) tuple
                    torch.cuda.synchronize()
                    forward_times.append((time.perf_counter() - start_time) * 1000)
            
            avg_forward_time = np.mean(forward_times)
            
            # Test training step
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.hparams.lr)
            
            # Create simplified loss (just MSE for testing)
            loss_fn = nn.MSELoss()
            
            training_times = []
            # Get device index safely  
            device_idx = getattr(device, 'index', 0) if hasattr(device, 'index') else 0
            start_metrics = self.get_gpu_metrics(device_idx)
            
            if cfg.general.use_amp:
                scaler = torch.cuda.amp.GradScaler()
                
            for _ in range(5):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                try:
                    if cfg.general.use_amp:
                        with torch.cuda.amp.autocast():
                            output = model(x)
                            # Model returns (loss, pred, mask) tuple
                            loss_val, pred, mask = output
                            # Use model's internal loss instead of external loss function
                            loss = loss_val
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(x)
                        # Model returns (loss, pred, mask) tuple
                        loss_val, pred, mask = output
                        # Use model's internal loss instead of external loss function
                        loss = loss_val
                        loss.backward()
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    training_times.append((time.perf_counter() - start_time) * 1000)
                    
                except Exception as e:
                    return {'error': f'Training step failed: {str(e)}'}
            
            # Get device index safely
            device_idx = getattr(device, 'index', 0) if hasattr(device, 'index') else 0
            end_metrics = self.get_gpu_metrics(device_idx)
            avg_training_time = np.mean(training_times)
            
            return {
                'forward_time_ms': avg_forward_time,
                'training_step_time_ms': avg_training_time,
                'gpu_util': end_metrics.get('gpu_util', 0),
                'memory_gb': end_metrics.get('memory_used_gb', 0),
                'power_w': end_metrics.get('power_w', 0)
            }
            
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()  # Clear cache after OOM
            print(f"  GPU OUT OF MEMORY: batch_size={batch_size}, spatial_size={cfg.model.model.img_size}")
            return {
                'error': 'CUDA_OOM',
                'error_msg': f'GPU out of memory with batch_size={batch_size}',
                'forward_time_ms': 0,
                'training_step_time_ms': 0,
                'gpu_util': 0,
                'memory_gb': 0
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  GPU MEMORY ERROR: {str(e)[:100]}...")
                return {
                    'error': 'CUDA_MEMORY_ERROR',
                    'error_msg': str(e)[:200],
                    'forward_time_ms': 0,
                    'training_step_time_ms': 0,
                    'gpu_util': 0,
                    'memory_gb': 0
                }
            else:
                # Other runtime errors (dimension mismatches, etc.)
                print(f"  MODEL ERROR: {str(e)[:100]}...")
                return {
                    'error': 'MODEL_ERROR',
                    'error_msg': str(e)[:200],
                    'forward_time_ms': 0,
                    'training_step_time_ms': 0,
                    'gpu_util': 0,
                    'memory_gb': 0
                }
                
        except Exception as e:
            print(f"  UNEXPECTED ERROR: {str(e)[:100]}...")
            return {
                'error': 'UNKNOWN_ERROR',
                'error_msg': str(e)[:200],
                'forward_time_ms': 0,
                'training_step_time_ms': 0,
                'gpu_util': 0,
                'memory_gb': 0
            }
        
    def compare_configurations(self) -> Dict:
        """Compare 240x240 vs 500x500 configurations"""
        print("=== COMPREHENSIVE 240x240 vs 500x500 COMPARISON ===")
        
        # Open text file for logging
        text_report_path = self.output_dir / f'diagnostic_report_{self.timestamp}.txt'
        self.text_file = open(text_report_path, 'w')
        
        self.log_both("=== COMPREHENSIVE 240x240 vs 500x500 COMPARISON ===")
        self.log_both(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_both(f"Data path: {self.base_cfg.dataset.csv_path}")
        self.log_both(f"Base configuration: {self.base_cfg.hparams.batch_size} batch, {self.base_cfg.dataloader.num_workers} workers")
        self.log_both("")
        
        results_240 = self.test_configuration_matrix(240)
        results_500 = self.test_configuration_matrix(500)
        
        # Find best overall configuration
        all_results = {**results_240, **results_500}
        
        # Filter successful configs with model performance
        successful_configs = {k: v for k, v in all_results.items() 
                            if v.get('status') == 'SUCCESS' and 'forward_time_ms' in v and 'error' not in v}
        
        if successful_configs:
            # Sort by training throughput (samples per second considering full training time)
            def get_training_throughput(config_key):
                config = successful_configs[config_key]
                if config.get('training_step_time_ms', 0) > 0:
                    return (config['batch_size'] * 1000) / config['training_step_time_ms']
                return 0
            
            best_config_key = max(successful_configs.keys(), key=get_training_throughput)
            best_config = successful_configs[best_config_key]
            
            self.log_both(f"\n=== OVERALL BEST CONFIGURATION ===")
            self.log_both(f"Configuration: {best_config_key}")
            self.log_both(f"Spatial size: {best_config['spatial_size']}x{best_config['spatial_size']}")
            self.log_both(f"Workers: {best_config['num_workers']}")
            self.log_both(f"Batch size: {best_config['batch_size']}")
            self.log_both(f"Data loading: {best_config['samples_per_sec']:.0f} samples/sec")
            self.log_both(f"Forward pass: {best_config.get('forward_time_ms', 0):.1f}ms")
            self.log_both(f"Training step: {best_config.get('training_step_time_ms', 0):.1f}ms")
            self.log_both(f"Training throughput: {get_training_throughput(best_config_key):.1f} samples/sec")
            self.log_both(f"GPU utilization: {best_config.get('gpu_util', 0):.1f}%")
            self.log_both(f"Memory usage: {best_config.get('memory_gb', 0):.1f}GB")
            
            # Compare with current configuration
            current_key = f"{self.base_cfg.spatial_size}x{self.base_cfg.spatial_size}_w{self.base_cfg.dataloader.num_workers}_b{self.base_cfg.hparams.batch_size}"
            if current_key in successful_configs:
                current_config = successful_configs[current_key]
                current_throughput = get_training_throughput(current_key)
                best_throughput = get_training_throughput(best_config_key)
                
                improvement = (best_throughput / current_throughput - 1) * 100 if current_throughput > 0 else 0
                
                self.log_both(f"\n=== COMPARISON WITH CURRENT CONFIG ===")
                self.log_both(f"Current: {current_throughput:.1f} samples/sec")
                self.log_both(f"Optimal: {best_throughput:.1f} samples/sec")
                self.log_both(f"Potential speedup: {improvement:.1f}%")
            
        # Close text file
        if self.text_file:
            self.log_both(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.text_file.close()
        
        return {
            'results_240': results_240,
            'results_500': results_500,
            'best_config': best_config_key if successful_configs else None,
            'summary': self.generate_summary(all_results)
        }
        
    def generate_summary(self, all_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze results by spatial size
        results_240 = {k: v for k, v in all_results.items() if v.get('spatial_size') == 240}
        results_500 = {k: v for k, v in all_results.items() if v.get('spatial_size') == 500}
        
        successful_240 = [v for v in results_240.values() if v.get('status') == 'SUCCESS']
        successful_500 = [v for v in results_500.values() if v.get('status') == 'SUCCESS']
        
        if successful_240 and successful_500:
            avg_throughput_240 = np.mean([v['samples_per_sec'] for v in successful_240])
            avg_throughput_500 = np.mean([v['samples_per_sec'] for v in successful_500])
            
            if avg_throughput_240 > avg_throughput_500 * 1.2:
                recommendations.append("240x240 images provide significantly better data loading performance")
            elif avg_throughput_500 > avg_throughput_240 * 1.2:
                recommendations.append("500x500 images provide better performance despite larger size")
            else:
                recommendations.append("Both spatial sizes show similar performance")
        
        # Analyze I/O bottlenecks
        io_bound_configs = [v for v in all_results.values() if v.get('io_overhead_pct', 0) > 70]
        if len(io_bound_configs) > len(all_results) * 0.5:
            recommendations.append("F: drive I/O is a major bottleneck - consider faster storage or caching")
        
        # Analyze worker patterns
        worker_performance = {}
        for v in all_results.values():
            if v.get('status') == 'SUCCESS':
                workers = v.get('num_workers', 0)
                if workers not in worker_performance:
                    worker_performance[workers] = []
                worker_performance[workers].append(v['samples_per_sec'])
        
        if worker_performance:
            avg_by_workers = {w: np.mean(speeds) for w, speeds in worker_performance.items()}
            optimal_workers = max(avg_by_workers.keys(), key=lambda w: avg_by_workers[w])
            recommendations.append(f"Optimal number of workers: {optimal_workers}")
        
        return recommendations


@hydra.main(version_base="1.3", config_path="model_training/conf", config_name="config")
def main(cfg: DictConfig):
    """Run comprehensive diagnostic comparing 240x240 vs 500x500"""
    
    print("=== COMPREHENSIVE SPATIAL SIZE COMPARISON ===")
    print(f"Testing both 240x240 and 500x500 configurations")
    print(f"Base batch size: {cfg.hparams.batch_size}")
    print(f"Base workers: {cfg.dataloader.num_workers}")
    print(f"Device: cuda:{cfg.general.device_id}")
    print(f"Parallel: {cfg.general.parallel.use_parallel}")
    print(f"AMP: {cfg.general.use_amp}")
    
    diag = ComprehensiveBottleneckDiagnostic(cfg)
    
    # Run comprehensive comparison
    results = diag.compare_configurations()
    
    # Save results
    report_path = diag.output_dir / f'comprehensive_comparison_{diag.timestamp}.json'
    text_report_path = diag.output_dir / f'diagnostic_report_{diag.timestamp}.txt'
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n=== RECOMMENDATIONS ===")
    for rec in results['summary']:
        print(f"• {rec}")
    
    print(f"\n=== RESULTS SAVED ===")
    print(f"Text report: {text_report_path}")
    print(f"JSON data: {report_path}")
    
    return results


if __name__ == '__main__':
    mp.freeze_support()
    main()
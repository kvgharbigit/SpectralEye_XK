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
            
    def find_max_batch_size(self, cfg: DictConfig, spatial_size: int, model_name: str) -> int:
        """Find maximum batch size that fits in GPU memory for given model/size"""
        self.log_both(f"  Finding maximum batch size for {model_name} at {spatial_size}x{spatial_size}...")
        
        max_batch = 1
        test_batches = [1, 2, 4, 6, 8, 12, 16, 24, 32]  # Test up to batch 32
        
        for batch_size in test_batches:
            try:
                # Quick memory test with minimal overhead
                torch.cuda.empty_cache()
                model_results = self.test_model_performance(cfg, batch_size, num_workers=1, quick_test=True)
                
                if 'error' not in model_results:
                    max_batch = batch_size
                    memory_used = model_results.get('memory_gb', 0)
                    self.log_both(f"    Batch {batch_size}: OK ({memory_used:.1f}GB)")
                else:
                    # Stop at first OOM
                    self.log_both(f"    Batch {batch_size}: OOM - Max batch size is {max_batch}")
                    break
                    
            except Exception as e:
                if "out of memory" in str(e).lower():
                    self.log_both(f"    Batch {batch_size}: OOM - Max batch size is {max_batch}")
                    break
                else:
                    # Other error, continue testing
                    continue
        
        self.log_both(f"  Maximum batch size for {model_name} at {spatial_size}x{spatial_size}: {max_batch}")
        return max_batch
    
    def test_configuration_matrix(self, spatial_size: int) -> Dict:
        """Test comprehensive matrix of configurations for given spatial size"""
        all_results = {}
        
        # Test both mae_small and mae_medium
        model_configs = [
            ('mae_small_240' if spatial_size == 240 else 'mae_small', 'mae_small'),
            ('mae_medium_240' if spatial_size == 240 else 'mae_medium', 'mae_medium')
        ]
        
        for config_file, model_name in model_configs:
            self.log_both(f"\n=== TESTING {spatial_size}x{spatial_size} - {model_name.upper()} ===")
            
            # Create configuration for this spatial size and model
            from copy import deepcopy
            cfg = OmegaConf.create(OmegaConf.to_yaml(self.base_cfg))
            cfg.spatial_size = spatial_size
            
            # Load appropriate model config
            config_path = f'model_training/conf/model/{config_file}.yaml'
            model_config = OmegaConf.load(config_path)
            cfg.model = model_config
            
            self.log_both(f"Model config: {cfg.model.name}")
            self.log_both(f"Image size: {cfg.model.model.img_size}")
            self.log_both(f"Spatial patch size: {cfg.model.model.spatial_patch_size}")
            self.log_both(f"Wavelength patches: {cfg.model.model.num_wavelengths}")
            
            # Find maximum batch size for this configuration
            max_batch = self.find_max_batch_size(cfg, spatial_size, model_name)
            
            # Test different configurations up to max batch
            worker_configs = [1, 2, 4]
            # Generate batch sizes up to maximum found
            batch_sizes = [b for b in [1, 2, 4, 6, 8, 12, 16, 24, 32] if b <= max_batch]
        
            results = {}
            
            # Test all permutations for this model
            self.log_both(f"\n=== MODEL PERFORMANCE TESTING ({spatial_size}x{spatial_size}) - {model_name.upper()} - ALL CONFIGURATIONS ===")
            self.log_both(f"Testing batch sizes: {batch_sizes}")
            self.log_both(f"{'Config':<20} {'Data Load':<12} {'Model FWD':<12} {'Model BWD':<12} {'GPU Util':<10} {'Memory':<10} {'Training/s':<12} {'Epoch(34k)':<12}")
            self.log_both("-" * 112)
        
            # Test model performance for all configurations directly
            for batch_size in batch_sizes:
                for num_workers in worker_configs:
                    config_key = f"{spatial_size}x{spatial_size}_{model_name}_w{num_workers}_b{batch_size}"
                
                    try:
                        # Test model performance directly (skip if batch > max_batch)
                        if batch_size > max_batch:
                            continue
                            
                        self.log_both(f"  Testing w{num_workers}_b{batch_size}...")
                        model_results = self.test_model_performance(cfg, batch_size, num_workers)
                        
                        results[config_key] = {
                            'spatial_size': spatial_size,
                            'model_name': model_name,
                            'batch_size': batch_size,
                            'num_workers': num_workers
                        }
                        results[config_key].update(model_results)
                        
                        config_display = f"w{num_workers}_b{batch_size}"
                        
                        if 'forward_time_ms' in model_results:
                            data_load_time = model_results.get('data_loading_time_ms', 0)
                            fwd_time = model_results['forward_time_ms']
                            bwd_time = model_results.get('training_step_time_ms', 0) - fwd_time
                            gpu_util = model_results.get('gpu_util', 0)
                            memory_gb = model_results.get('memory_gb', 0)
                            
                            # Calculate overall training throughput (samples per second in training)
                            if model_results.get('training_step_time_ms', 0) > 0:
                                training_samples_per_sec = (batch_size * 1000) / model_results['training_step_time_ms']
                                # Calculate epoch time for 34k dataset
                                epoch_seconds = 34000 / training_samples_per_sec
                                epoch_minutes = epoch_seconds / 60
                                if epoch_minutes >= 60:
                                    epoch_time = f"{epoch_minutes/60:.1f}h"
                                else:
                                    epoch_time = f"{epoch_minutes:.0f}min"
                            else:
                                training_samples_per_sec = 0
                                epoch_time = "N/A"
                            
                            self.log_both(f"{config_display:<20} {data_load_time:<12.1f} {fwd_time:<12.1f} {bwd_time:<12.1f} {gpu_util:<10.1f} {memory_gb:<10.1f} {training_samples_per_sec:<12.1f} {epoch_time:<12}")
                        else:
                            self.log_both(f"{config_display:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12}")
                            
                    except Exception as e:
                        results[config_key] = {
                            'spatial_size': spatial_size,
                            'model_name': model_name,
                            'batch_size': batch_size,
                            'num_workers': num_workers,
                            'status': f'ERROR: {str(e)[:30]}'
                        }
                        self.log_both(f"w{num_workers}_b{batch_size}        ERROR: {str(e)[:50]}")
            
            # Add this model's results to all_results
            all_results[f"{spatial_size}_{model_name}"] = results
        
        return all_results
        
    def test_model_performance(self, cfg: DictConfig, batch_size: int, num_workers: int, quick_test: bool = False) -> Dict:
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
            
            # Create REAL DataLoader with actual SpectralEye data pipeline
            img_size = cfg.model.model.img_size
            
            if not quick_test:
                print(f"DEBUG - Setting up real data pipeline:")
                print(f"  Loading from: {cfg.dataset.csv_path}")
                print(f"  Image size: {img_size}")
                print(f"  Batch size: {batch_size}")
                print(f"  Workers: {num_workers}")
            
            # Import actual dataset and preprocessing
            from model_training.dataset.combined_dataset import get_dataset
            from model_training.utils.preprocess_hsi import preprocess_hsi
            
            # Create real dataset (small subset for testing)
            train_dataset, val_dataset = get_dataset(
                csv_path=cfg.dataset.csv_path,
                train_ratio=cfg.dataset.train_ratio,
                seed=cfg.dataset.seed,
                trial_mode=True,  # Use small subset for testing
                trial_size=50,    # Very small for performance testing
                transform=None    # Skip augmentations for cleaner benchmarking
            )
            
            # Create real DataLoader with actual configuration
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=num_workers > 1,
                shuffle=False,    # Disable shuffle for consistent timing
                timeout=60 if num_workers > 0 else 0
            )
            
            # Get a real batch from F: drive with timing
            if not quick_test:
                print(f"  Loading real batch from F: drive...")
            
            batch_load_start = time.perf_counter()
            real_batch = None
            
            for i, batch in enumerate(dataloader):
                if i >= 1:  # Just need one batch for model testing
                    break
                    
                # Unpack SpectralEye batch format: hs_cube, label, rgb
                hs_cube, label, rgb = batch
                
                # Apply real preprocessing pipeline (log transform + normalization)
                hs_cube = preprocess_hsi(hs_cube)
                
                # Transfer to GPU (as done in actual training)
                hs_cube = hs_cube.to(device, non_blocking=True)
                
                real_batch = hs_cube
                torch.cuda.synchronize()
                break
            
            data_load_time = (time.perf_counter() - batch_load_start) * 1000
            
            if real_batch is None:
                return {'error': 'No data batches loaded from F: drive'}
                
            # Use real data instead of synthetic
            x = real_batch
            
            if not quick_test:
                print(f"  Real data loaded: {x.shape}, F: drive + preprocessing time: {data_load_time:.1f}ms")
            
            model.eval()
            
            # Test forward pass
            forward_times = []
            with torch.no_grad():
                # Warmup (fewer iterations for quick test)
                warmup_iterations = 1 if quick_test else 3
                for i in range(warmup_iterations):
                    try:
                        if not quick_test:
                            print(f"  Warmup {i+1}/{warmup_iterations}...")
                        output = model(x)
                    # Model returns (loss, pred, mask) tuple
                        if isinstance(output, tuple) and not quick_test:
                            print(f"  Success! Output tuple with {len(output)} elements")
                        elif not quick_test:
                            print(f"  Success! Output shape: {output.shape}")
                    except Exception as e:
                        if not quick_test:
                            print(f"  Forward pass failed: {str(e)}")
                        return {'error': f'Forward pass failed: {str(e)}'}
                        
                # Benchmark forward (fewer iterations for quick test)
                benchmark_iterations = 2 if quick_test else 10
                if not quick_test:
                    print("  Running forward benchmark...")
                torch.cuda.synchronize()
                for _ in range(benchmark_iterations):
                    start_time = time.perf_counter()
                    output = model(x)
                    # Model returns (loss, pred, mask) tuple
                    torch.cuda.synchronize()
                    forward_times.append((time.perf_counter() - start_time) * 1000)
            
            avg_forward_time = np.mean(forward_times)
            if not quick_test:
                print(f"  Forward pass completed: {len(forward_times)} runs, avg time: {avg_forward_time:.1f}ms")
            
            # Skip training test for quick memory check
            if quick_test:
                # Get memory usage and return early
                device_idx = getattr(device, 'index', 0) if hasattr(device, 'index') else 0
                end_metrics = self.get_gpu_metrics(device_idx)
                return {
                    'forward_time_ms': avg_forward_time,
                    'data_loading_time_ms': data_load_time,
                    'memory_gb': end_metrics.get('memory_used_gb', 0),
                }
            
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
                            # Ensure loss is scalar for backprop
                            loss = torch.mean(loss_val) if loss_val.dim() > 0 else loss_val
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(x)
                        # Model returns (loss, pred, mask) tuple
                        loss_val, pred, mask = output
                        # Use model's internal loss instead of external loss function
                        # Ensure loss is scalar for backprop
                        loss = torch.mean(loss_val) if loss_val.dim() > 0 else loss_val
                        loss.backward()
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    training_times.append((time.perf_counter() - start_time) * 1000)
                    
                except Exception as e:
                    print(f"  Training step failed: {str(e)}")
                    return {'error': f'Training step failed: {str(e)}'}
            
            # Get device index safely
            device_idx = getattr(device, 'index', 0) if hasattr(device, 'index') else 0
            end_metrics = self.get_gpu_metrics(device_idx)
            avg_training_time = np.mean(training_times)
            
            print(f"  Training completed: {len(training_times)} steps, avg time: {avg_training_time:.1f}ms")
            
            return {
                'forward_time_ms': avg_forward_time,
                'training_step_time_ms': avg_training_time,
                'data_loading_time_ms': data_load_time,
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
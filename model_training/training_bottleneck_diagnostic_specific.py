#!/usr/bin/env python3
"""
Specific Training Bottleneck Diagnostic for YOUR SpectralEye Model

This script specifically tests your exact training configuration to identify bottlenecks.
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
import psutil
import pynvml
import matplotlib.pyplot as plt
from datetime import datetime
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py
from typing import Dict, List, Optional
import pandas as pd
import multiprocessing as mp
from contextlib import contextmanager

# Import your actual model and dataset
from model_training.models.spectral_gpt.spectral_gpt import MaskedAutoencoderViT
from model_training.dataset.combined_dataset import SegmentationDataset, get_dataset
from model_training.losses.custom_loss import CustomLoss


class SpecificBottleneckDiagnostic:
    """Diagnostic class specifically for your training setup"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.general.device_id}" if cfg.general.use_cuda else "cpu")
        self.output_dir = Path('diagnostic_results_specific')
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize GPU monitoring
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(cfg.general.device_id)
        
        self.results = {}
        
    def get_gpu_metrics(self) -> Dict:
        """Get current GPU metrics"""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            
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
            
    @contextmanager
    def profile_section(self, name: str):
        """Profile a code section"""
        torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        start_metrics = self.get_gpu_metrics()
        
        yield
        
        torch.cuda.synchronize(self.device)
        end_time = time.perf_counter()
        end_metrics = self.get_gpu_metrics()
        
        self.results[name] = {
            'duration': end_time - start_time,
            'gpu_metrics_start': start_metrics,
            'gpu_metrics_end': end_metrics
        }
        
    def test_actual_data_loading(self) -> Dict:
        """Test your actual data loading pipeline"""
        print("\n=== Testing YOUR Actual Data Loading Pipeline ===")
        print(f"Data path: {self.cfg.dataset.csv_path}")
        print(f"Batch size: {self.cfg.hparams.batch_size}")
        print(f"Workers: {self.cfg.dataloader.num_workers}")
        
        # Create your actual dataset
        train_dataset, val_dataset = get_dataset(
            csv_path=self.cfg.dataset.csv_path,
            train_ratio=self.cfg.dataset.train_ratio,
            seed=self.cfg.dataset.seed,
            trial_mode=True,  # Use trial mode for faster testing
            trial_size=100,
            transform=None
        )
        
        print(f"Dataset size (trial): {len(train_dataset)}")
        
        # Test different number of workers
        results = {}
        worker_configs = [0, 1, 2, 4, 8] if os.name == 'nt' else [0, 1, 2, 4, 8, 16]
        
        for num_workers in worker_configs:
            if num_workers == 0 and os.name == 'nt':
                continue  # Skip 0 workers on Windows
                
            print(f"\nTesting with {num_workers} workers")
            
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.hparams.batch_size,
                num_workers=num_workers,
                pin_memory=self.cfg.dataloader.pin_memory and num_workers > 0,
                prefetch_factor=self.cfg.dataloader.prefetch_factor if num_workers > 0 else None,
                persistent_workers=self.cfg.dataloader.persistent_workers and num_workers > 0,
                shuffle=self.cfg.dataloader.shuffle
            )
            
            # Warmup
            warmup_batches = 5
            for i, batch in enumerate(dataloader):
                if i >= warmup_batches:
                    break
                    
            # Benchmark
            with self.profile_section(f'dataload_{num_workers}_workers'):
                num_batches = 20
                batch_times = []
                transfer_times = []
                
                for i, batch in enumerate(dataloader):
                    if i >= num_batches:
                        break
                        
                    batch_start = time.perf_counter()
                    
                    # Get data
                    if isinstance(batch, dict):
                        spectral = batch['spectral']
                        label = batch.get('label', None)
                    else:
                        spectral = batch[0]
                        label = batch[1] if len(batch) > 1 else None
                        
                    # Transfer to GPU (this is what happens in training)
                    transfer_start = time.perf_counter()
                    spectral_gpu = spectral.to(self.device, non_blocking=True)
                    if label is not None and hasattr(label, 'to'):
                        label_gpu = label.to(self.device, non_blocking=True)
                    torch.cuda.synchronize()
                    transfer_time = time.perf_counter() - transfer_start
                    
                    batch_time = time.perf_counter() - batch_start
                    batch_times.append(batch_time)
                    transfer_times.append(transfer_time)
                    
            avg_batch_time = np.mean(batch_times)
            avg_transfer_time = np.mean(transfer_times)
            samples_per_sec = (num_batches * self.cfg.hparams.batch_size) / sum(batch_times)
            
            results[num_workers] = {
                'avg_batch_time_ms': avg_batch_time * 1000,
                'avg_transfer_time_ms': avg_transfer_time * 1000,
                'samples_per_sec': samples_per_sec,
                'transfer_overhead_pct': (avg_transfer_time / avg_batch_time) * 100
            }
            
            print(f"  Avg batch time: {avg_batch_time*1000:.1f}ms")
            print(f"  Avg transfer time: {avg_transfer_time*1000:.1f}ms ({results[num_workers]['transfer_overhead_pct']:.1f}%)")
            print(f"  Samples/sec: {samples_per_sec:.1f}")
            
        return results
        
    def test_actual_model_forward(self) -> Dict:
        """Test your actual model forward pass"""
        print("\n=== Testing YOUR Model (MAE Medium) ===")
        print(f"Model: {self.cfg.model.name}")
        print(f"Image size: {self.cfg.model.model.img_size}")
        print(f"Batch size: {self.cfg.hparams.batch_size}")
        
        # Create your actual model
        model = MaskedAutoencoderViT(
            img_size=self.cfg.model.model.img_size,
            num_channels=self.cfg.model.model.num_channels,
            num_wavelengths=self.cfg.model.model.num_wavelengths,
            spatial_patch_size=self.cfg.model.model.spatial_patch_size,
            wavelength_patch_size=self.cfg.model.model.wavelength_patch_size,
            encoder_embed_dim=self.cfg.model.model.encoder_embed_dim,
            encoder_depth=self.cfg.model.model.encoder_depth,
            encoder_num_heads=self.cfg.model.model.encoder_num_heads,
            decoder_embed_dim=self.cfg.model.model.decoder_embed_dim,
            decoder_depth=self.cfg.model.model.decoder_depth,
            decoder_num_heads=self.cfg.model.model.decoder_num_heads,
            mlp_ratio=self.cfg.model.model.mlp_ratio,
            mask_ratio=self.cfg.model.model.mask_ratio
        ).to(self.device)
        
        # Use DataParallel if configured
        if self.cfg.general.parallel.use_parallel:
            # For DataParallel, model must be on device 0 first
            model = model.to('cuda:0')
            model = nn.DataParallel(model, device_ids=self.cfg.general.parallel.device_ids)
            print(f"Using DataParallel on devices: {self.cfg.general.parallel.device_ids}")
            # Update device for input tensors
            self.device = torch.device('cuda:0')
            
        model.eval()
        
        # Create input tensor matching your data
        batch_size = self.cfg.hparams.batch_size
        img_size = self.cfg.model.model.img_size
        num_wavelengths = self.cfg.model.model.num_wavelengths
        
        # Your data shape: [B, H, W, C] where C=wavelengths
        x = torch.randn(batch_size, img_size, img_size, num_wavelengths, device=self.device)
        
        # Test forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = model(x)
                torch.cuda.synchronize()
                
            # Benchmark
            with self.profile_section('forward_pass'):
                iterations = 20
                forward_times = []
                
                for _ in range(iterations):
                    iter_start = time.perf_counter()
                    output = model(x)
                    torch.cuda.synchronize()
                    forward_times.append(time.perf_counter() - iter_start)
                    
        avg_forward_time = np.mean(forward_times)
        
        # Test with mixed precision
        if self.cfg.general.use_amp:
            print("\nTesting with AMP (mixed precision)...")
            scaler = torch.cuda.amp.GradScaler()
            
            with self.profile_section('forward_pass_amp'):
                amp_times = []
                
                for _ in range(iterations):
                    iter_start = time.perf_counter()
                    with torch.cuda.amp.autocast():
                        output = model(x)
                    torch.cuda.synchronize()
                    amp_times.append(time.perf_counter() - iter_start)
                    
            avg_amp_time = np.mean(amp_times)
        else:
            avg_amp_time = None
            
        # Test full training step
        print("\nTesting full training step...")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.hparams.lr)
        loss_fn = CustomLoss(
            reconstruction_weight=self.cfg.loss.reconstruction_weight,
            angle_weight=self.cfg.loss.angle_weight,
            variance_weight=self.cfg.loss.variance_weight,
            range_weight=self.cfg.loss.range_weight
        ).to(self.device)
        
        with self.profile_section('training_step'):
            train_times = []
            
            for _ in range(10):
                iter_start = time.perf_counter()
                
                if self.cfg.general.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(x)
                        # CustomLoss expects (reconstructed, original, latent, rgb)
                        # For MAE, output is the reconstructed image
                        dummy_latent = torch.randn_like(x[:, :, :, :3]).to(self.device)  # Dummy latent
                        dummy_rgb = torch.randn(x.shape[0], x.shape[1], x.shape[2], 3).to(self.device)  # Dummy RGB
                        loss = loss_fn(output, x, dummy_latent, dummy_rgb)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(x)
                    dummy_latent = torch.randn_like(x[:, :, :, :3]).to(self.device)
                    dummy_rgb = torch.randn(x.shape[0], x.shape[1], x.shape[2], 3).to(self.device)
                    loss = loss_fn(output, x, dummy_latent, dummy_rgb)
                    loss.backward()
                    optimizer.step()
                    
                optimizer.zero_grad()
                torch.cuda.synchronize()
                
                train_times.append(time.perf_counter() - iter_start)
                
        avg_train_time = np.mean(train_times)
        
        results = {
            'forward_time_ms': avg_forward_time * 1000,
            'forward_time_amp_ms': avg_amp_time * 1000 if avg_amp_time else None,
            'training_step_time_ms': avg_train_time * 1000,
            'amp_speedup': avg_forward_time / avg_amp_time if avg_amp_time else None,
            'gpu_metrics': self.results['forward_pass']['gpu_metrics_end']
        }
        
        print(f"\nResults:")
        print(f"  Forward pass: {results['forward_time_ms']:.1f}ms")
        if results['forward_time_amp_ms']:
            print(f"  Forward pass (AMP): {results['forward_time_amp_ms']:.1f}ms")
            print(f"  AMP speedup: {results['amp_speedup']:.2f}x")
        print(f"  Full training step: {results['training_step_time_ms']:.1f}ms")
        print(f"  GPU utilization: {results['gpu_metrics'].get('gpu_util', 0):.1f}%")
        print(f"  GPU memory used: {results['gpu_metrics'].get('memory_used_gb', 0):.1f}GB")
        
        return results
        
    def test_full_epoch(self) -> Dict:
        """Test a full epoch to identify bottlenecks"""
        print("\n=== Testing Full Training Epoch ===")
        
        # Create dataset and dataloader
        train_dataset, val_dataset = get_dataset(
            csv_path=self.cfg.dataset.csv_path,
            train_ratio=self.cfg.dataset.train_ratio,
            seed=self.cfg.dataset.seed,
            trial_mode=True,
            trial_size=50,  # Small size for testing
            transform=None
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.hparams.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
            prefetch_factor=self.cfg.dataloader.prefetch_factor,
            persistent_workers=self.cfg.dataloader.persistent_workers,
            shuffle=self.cfg.dataloader.shuffle
        )
        
        # Create model
        model = MaskedAutoencoderViT(
            img_size=self.cfg.model.model.img_size,
            num_channels=self.cfg.model.model.num_channels,
            num_wavelengths=self.cfg.model.model.num_wavelengths,
            spatial_patch_size=self.cfg.model.model.spatial_patch_size,
            wavelength_patch_size=self.cfg.model.model.wavelength_patch_size,
            encoder_embed_dim=self.cfg.model.model.encoder_embed_dim,
            encoder_depth=self.cfg.model.model.encoder_depth,
            encoder_num_heads=self.cfg.model.model.encoder_num_heads,
            decoder_embed_dim=self.cfg.model.model.decoder_embed_dim,
            decoder_depth=self.cfg.model.model.decoder_depth,
            decoder_num_heads=self.cfg.model.model.decoder_num_heads,
            mlp_ratio=self.cfg.model.model.mlp_ratio,
            mask_ratio=self.cfg.model.model.mask_ratio
        ).to(self.device)
        
        if self.cfg.general.parallel.use_parallel:
            # For DataParallel, model must be on device 0 first
            model = model.to('cuda:0')
            model = nn.DataParallel(model, device_ids=self.cfg.general.parallel.device_ids)
            # Update device for tensors
            original_device = self.device
            self.device = torch.device('cuda:0')
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.hparams.lr)
        loss_fn = CustomLoss(
            reconstruction_weight=self.cfg.loss.reconstruction_weight,
            angle_weight=self.cfg.loss.angle_weight,
            variance_weight=self.cfg.loss.variance_weight,
            range_weight=self.cfg.loss.range_weight
        ).to(self.device)
        
        # Profile different parts of the epoch
        epoch_timings = {
            'data_loading': [],
            'forward_pass': [],
            'loss_computation': [],
            'backward_pass': [],
            'optimizer_step': [],
            'total_batch': []
        }
        
        print("Running mini epoch...")
        model.train()
        
        for i, batch in enumerate(train_loader):
            if i >= 10:  # Only test 10 batches
                break
                
            batch_start = time.perf_counter()
            
            # Data loading time (already measured)
            if isinstance(batch, dict):
                spectral = batch['spectral'].to(self.device, non_blocking=True)
                label = batch.get('label', None)
                if label is not None and hasattr(label, 'to'):
                    label = label.to(self.device, non_blocking=True)
            else:
                spectral = batch[0].to(self.device, non_blocking=True)
                if len(batch) > 1 and hasattr(batch[1], 'to'):
                    label = batch[1].to(self.device, non_blocking=True)
                else:
                    label = None
            
            torch.cuda.synchronize()
            data_time = time.perf_counter() - batch_start
            epoch_timings['data_loading'].append(data_time)
            
            # Forward pass
            forward_start = time.perf_counter()
            if self.cfg.general.use_amp:
                with torch.cuda.amp.autocast():
                    output = model(spectral)
            else:
                output = model(spectral)
            torch.cuda.synchronize()
            forward_time = time.perf_counter() - forward_start
            epoch_timings['forward_pass'].append(forward_time)
            
            # Loss computation
            loss_start = time.perf_counter()
            if self.cfg.general.use_amp:
                with torch.cuda.amp.autocast():
                    dummy_latent = torch.randn_like(spectral[:, :, :, :3]).to(self.device)
                    dummy_rgb = torch.randn(spectral.shape[0], spectral.shape[1], spectral.shape[2], 3).to(self.device)
                    loss = loss_fn(output, spectral, dummy_latent, dummy_rgb)
            else:
                dummy_latent = torch.randn_like(spectral[:, :, :, :3]).to(self.device)
                dummy_rgb = torch.randn(spectral.shape[0], spectral.shape[1], spectral.shape[2], 3).to(self.device)
                loss = loss_fn(output, spectral, dummy_latent, dummy_rgb)
            torch.cuda.synchronize()
            loss_time = time.perf_counter() - loss_start
            epoch_timings['loss_computation'].append(loss_time)
            
            # Backward pass
            backward_start = time.perf_counter()
            if self.cfg.general.use_amp:
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.cuda.synchronize()
            backward_time = time.perf_counter() - backward_start
            epoch_timings['backward_pass'].append(backward_time)
            
            # Optimizer step
            opt_start = time.perf_counter()
            if self.cfg.general.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            opt_time = time.perf_counter() - opt_start
            epoch_timings['optimizer_step'].append(opt_time)
            
            total_time = time.perf_counter() - batch_start
            epoch_timings['total_batch'].append(total_time)
            
        # Analyze results
        results = {}
        for component, times in epoch_timings.items():
            if times:
                avg_time = np.mean(times) * 1000  # Convert to ms
                pct_of_total = (np.mean(times) / np.mean(epoch_timings['total_batch'])) * 100 if component != 'total_batch' else 100
                results[component] = {
                    'avg_time_ms': avg_time,
                    'pct_of_total': pct_of_total
                }
                
        print("\nEpoch timing breakdown:")
        for component, stats in results.items():
            if component != 'total_batch':
                print(f"  {component}: {stats['avg_time_ms']:.1f}ms ({stats['pct_of_total']:.1f}%)")
        print(f"  Total per batch: {results['total_batch']['avg_time_ms']:.1f}ms")
        
        return results
        
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n=== Generating Diagnostic Report ===")
        
        report = {
            'timestamp': self.timestamp,
            'configuration': OmegaConf.to_container(self.cfg),
            'device_info': {
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name(self.device),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            },
            'results': self.results,
            'bottleneck_analysis': self._analyze_bottlenecks()
        }
        
        # Save report
        report_path = self.output_dir / f'specific_diagnostic_{self.timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\nReport saved to: {report_path}")
        
        # Print summary
        print("\n=== BOTTLENECK ANALYSIS ===")
        for bottleneck in report['bottleneck_analysis']:
            print(f"\n{bottleneck}")
            
        return report
        
    def _analyze_bottlenecks(self) -> List[str]:
        """Analyze results and identify bottlenecks"""
        bottlenecks = []
        
        # Analyze data loading
        if 'dataload_2_workers' in self.results:
            current_config_time = self.results.get(f'dataload_{self.cfg.dataloader.num_workers}_workers', {}).get('duration', 0)
            best_config = min([(k, v['duration']) for k, v in self.results.items() if 'dataload_' in k], key=lambda x: x[1])
            
            if current_config_time > best_config[1] * 1.2:
                optimal_workers = int(best_config[0].split('_')[1])
                bottlenecks.append(f"DATA LOADING: Current config uses {self.cfg.dataloader.num_workers} workers, "
                                 f"but {optimal_workers} workers would be {current_config_time/best_config[1]:.1f}x faster")
                
        # Analyze GPU utilization
        gpu_utils = []
        for result in self.results.values():
            if isinstance(result, dict) and 'gpu_metrics_end' in result:
                gpu_util = result['gpu_metrics_end'].get('gpu_util', 0)
                if gpu_util > 0:
                    gpu_utils.append(gpu_util)
                    
        if gpu_utils:
            avg_gpu_util = np.mean(gpu_utils)
            if avg_gpu_util < 80:
                bottlenecks.append(f"LOW GPU UTILIZATION: Average {avg_gpu_util:.1f}%. Consider:")
                bottlenecks.append("  - Increasing batch size (current: {})".format(self.cfg.hparams.batch_size))
                bottlenecks.append("  - Optimizing data loading pipeline")
                bottlenecks.append("  - Using gradient accumulation for larger effective batch size")
                
        # Analyze memory usage
        mem_used = []
        for result in self.results.values():
            if isinstance(result, dict) and 'gpu_metrics_end' in result:
                mem = result['gpu_metrics_end'].get('memory_used_gb', 0)
                if mem > 0:
                    mem_used.append(mem)
                    
        if mem_used:
            max_mem = max(mem_used)
            total_mem = self.results.get('forward_pass', {}).get('gpu_metrics_end', {}).get('memory_free_gb', 11) + max_mem
            mem_utilization = (max_mem / total_mem) * 100
            
            if mem_utilization < 70:
                bottlenecks.append(f"LOW MEMORY UTILIZATION: Using only {max_mem:.1f}GB of {total_mem:.1f}GB ({mem_utilization:.1f}%)")
                bottlenecks.append(f"  - Consider increasing batch size from {self.cfg.hparams.batch_size}")
                
        # Analyze I/O vs compute balance
        if 'training_step' in self.results:
            train_time = self.results['training_step']['duration']
            # Rough estimate of data loading time per batch
            data_time = self.results.get(f'dataload_{self.cfg.dataloader.num_workers}_workers', {}).get('duration', 0) / 20
            
            if data_time > train_time * 0.2:
                bottlenecks.append(f"I/O BOUND: Data loading takes {data_time*1000:.1f}ms vs {train_time*1000:.1f}ms training")
                bottlenecks.append("  - Consider using more workers or optimizing data loading")
                bottlenecks.append("  - Check if F: drive is the bottleneck (network drive?)")
                
        return bottlenecks if bottlenecks else ["No major bottlenecks identified"]


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run specific diagnostic with your training configuration"""
    
    # Override some settings for diagnostic
    cfg.dataset.trial_mode = True
    cfg.dataset.trial_size = 100
    
    print("=== SpectralEye Training Bottleneck Diagnostic ===")
    print(f"Configuration: {cfg.model.name}")
    print(f"Batch size: {cfg.hparams.batch_size}")
    print(f"Workers: {cfg.dataloader.num_workers}")
    print(f"Device: cuda:{cfg.general.device_id}")
    print(f"Parallel: {cfg.general.parallel.use_parallel}")
    print(f"AMP: {cfg.general.use_amp}")
    
    diag = SpecificBottleneckDiagnostic(cfg)
    
    # Run tests
    diag.test_actual_data_loading()
    diag.test_actual_model_forward()
    diag.test_full_epoch()
    
    # Generate report
    report = diag.generate_report()
    
    return report


if __name__ == '__main__':
    mp.freeze_support()
    main()
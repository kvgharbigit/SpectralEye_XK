#!/usr/bin/env python3
"""
Simple batch size tester - runs each batch size in a separate Python process
to avoid memory accumulation issues.
"""
import os
import subprocess
import sys
from pathlib import Path

# Test configurations
test_configs = [
    ("mae_small_240", 1, [1, 2, 4, 8]),  # (model, workers, batch_sizes)
    ("mae_medium_240", 1, [1, 2, 4]),
]

def run_single_test(model_name, workers, batch_size, gpu_count):
    """Run a single configuration test in a separate process"""
    
    # Create a simple test script
    test_script = f"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["TORCH_DISTRIBUTED_USE_LIBUV"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import time
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose

def single_gpu_test(rank, world_size, model_name, batch_size, workers):
    # Setup DDP
    from torch.distributed import TCPStore
    from datetime import timedelta
    
    timeout = timedelta(seconds=300)
    if rank == 0:
        store = TCPStore("127.0.0.1", 29502, world_size=world_size, is_master=True, 
                       timeout=timeout, wait_for_workers=True, use_libuv=False)
    else:
        store = TCPStore("127.0.0.1", 29502, world_size=world_size, is_master=False, 
                       timeout=timeout, wait_for_workers=True, use_libuv=False)
    
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{{rank}}")
    
    try:
        # Load config
        with initialize(version_base=None, config_path="model_training/conf"):
            cfg = compose(config_name="full_run_240")
        
        # Update for this test
        cfg.model.name = model_name
        cfg.dataset.trial_mode = False
        cfg.hparams.batch_size = batch_size
        cfg.dataloader.num_workers = workers
        
        # Create dataset
        dataset_fn = instantiate(cfg.dataset)
        datasets = dataset_fn(transform=None)
        train_dataset = datasets[0]
        
        # Create sampler
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=False,
            sampler=sampler,
            drop_last=True
        )
        
        # Create model
        model = instantiate(cfg.model.model).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        
        # Preprocessor
        def preprocess_hsi(x):
            x = torch.log1p(x)
            x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-8)
            return x
        
        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 1:
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
            
        # Benchmark
        forward_times = []
        backward_times = []
        data_times = []
        
        torch.cuda.synchronize()
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Test 5 iterations
                break
                
            # Data loading time
            data_start = time.perf_counter()
            hs_cube, label, rgb = batch
            hs_cube = preprocess_hsi(hs_cube)
            hs_cube = hs_cube.to(device, non_blocking=True)
            torch.cuda.synchronize()
            data_times.append(time.perf_counter() - data_start)
            
            # Forward
            forward_start = time.perf_counter()
            output = model(hs_cube)
            if isinstance(output, tuple):
                loss = output[0]
            else:
                loss = output.mean()
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - forward_start)
            
            # Backward
            backward_start = time.perf_counter()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - backward_start)
        
        # Calculate metrics
        if rank == 0 and forward_times:
            avg_data_time = np.mean(data_times)
            avg_forward_time = np.mean(forward_times) * 1000
            avg_backward_time = np.mean(backward_times) * 1000
            data_rate = batch_size / avg_data_time
            samples_per_sec = batch_size / (avg_forward_time/1000 + avg_backward_time/1000)
            total_throughput = samples_per_sec * world_size
            
            print(f"{{model_name}}_{{world_size}}GPU_w{{workers}}_b{{batch_size}}: "
                  f"Data={{data_rate:.1f}}/s, "
                  f"Forward={{avg_forward_time:.1f}}ms, "
                  f"Backward={{avg_backward_time:.1f}}ms, "
                  f"Training={{samples_per_sec:.1f}}/s, "
                  f"Total={{total_throughput:.1f}}/s")
                  
    except Exception as e:
        if rank == 0:
            print(f"{{model_name}}_{{world_size}}GPU_w{{workers}}_b{{batch_size}}: ERROR - {{str(e)}}")
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(single_gpu_test, args=({gpu_count}, "{model_name}", {batch_size}, {workers}), nprocs={gpu_count}, join=True)
"""

    # Write test script to temporary file
    script_path = f"temp_test_{model_name}_{gpu_count}gpu_w{workers}_b{batch_size}.py"
    with open(script_path, 'w') as f:
        f.write(test_script)
    
    try:
        # Run the test in a separate Python process
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip() and "RequestsDependencyWarning" not in result.stderr:
            print(f"STDERR: {result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        print(f"{model_name}_{gpu_count}GPU_w{workers}_b{batch_size}: TIMEOUT")
    except Exception as e:
        print(f"{model_name}_{gpu_count}GPU_w{workers}_b{batch_size}: ERROR - {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(script_path):
            os.remove(script_path)

def main():
    print("=== BATCH SIZE OPTIMIZATION TEST ===")
    print("Testing different batch sizes with clean memory for each test")
    print()
    
    # Test different GPU counts
    for gpu_count in [1, 2, 3]:
        print(f"\\n=== TESTING WITH {gpu_count} GPU(s) ===")
        
        for model_name, workers, batch_sizes in test_configs:
            for batch_size in batch_sizes:
                run_single_test(model_name, workers, batch_size, gpu_count)
                time.sleep(2)  # Brief pause between tests
    
    print("\\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    import time
    main()
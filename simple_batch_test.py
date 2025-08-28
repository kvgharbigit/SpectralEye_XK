#!/usr/bin/env python3
"""
Simple batch size tester - runs each batch size in a separate Python process
to avoid memory accumulation issues.
"""
import os
import subprocess
import sys
from pathlib import Path

# Test configurations: (model, workers, batch_sizes, dataset_config)
test_configs = [
    ("mae_small_240", 1, [1, 2, 4, 6, 8, 12], "dataset_240"),
    ("mae_medium_240", 1, [1, 2, 4, 6], "dataset_240"),
    ("mae_small", 1, [1, 2, 4, 6, 8], "combined_dataset"), 
    ("mae_medium", 1, [1, 2, 4], "combined_dataset"),
]

def run_single_test(model_name, workers, batch_size, gpu_count, dataset_config):
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

def single_gpu_test(rank, world_size, model_name, batch_size, workers, dataset_config):
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
        # Load appropriate config based on dataset
        with initialize(version_base=None, config_path="model_training/conf"):
            if "240" in dataset_config:
                cfg = compose(config_name="full_run_240")
            else:
                cfg = compose(config_name="config")  # Default 500x500 config
        
        # Update for this test
        cfg.model.name = model_name
        cfg.dataset.trial_mode = False
        cfg.hparams.batch_size = batch_size
        cfg.dataloader.num_workers = workers
        
        # Override dataset if needed
        try:
            current_dataset = cfg.defaults.dataset[0] if hasattr(cfg, 'defaults') and hasattr(cfg.defaults, 'dataset') else None
        except:
            current_dataset = None
            
        if dataset_config != current_dataset:
            # Update dataset configuration
            with initialize(version_base=None, config_path="model_training/conf/dataset"):
                dataset_cfg = compose(config_name=dataset_config)
                cfg.dataset = dataset_cfg
        
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
            
            result_line = f"{{model_name}}_{{world_size}}GPU_w{{workers}}_b{{batch_size}}: Data={{data_rate:.1f}}/s, Forward={{avg_forward_time:.1f}}ms, Backward={{avg_backward_time:.1f}}ms, Training={{samples_per_sec:.1f}}/s, Total={{total_throughput:.1f}}/s"
            print(result_line)
                  
    except Exception as e:
        if rank == 0:
            print(f"{{model_name}}_{{world_size}}GPU_w{{workers}}_b{{batch_size}}: ERROR - {{str(e)}}")
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(single_gpu_test, args=({gpu_count}, "{model_name}", {batch_size}, {workers}, "{dataset_config}"), nprocs={gpu_count}, join=True)
"""

    # Write test script to temporary file
    script_path = f"temp_test_{model_name}_{gpu_count}gpu_w{workers}_b{batch_size}.py"
    with open(script_path, 'w') as f:
        f.write(test_script)
    
    try:
        # Run the test in a separate Python process
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        output = result.stdout.strip()
        error_output = result.stderr.strip()
        
        print(f"DEBUG - Return code: {result.returncode}")
        print(f"DEBUG - STDOUT: '{output}'")
        print(f"DEBUG - STDERR: '{error_output}'")
        
        if output:
            print(output)
            return output
        
        if error_output and "RequestsDependencyWarning" not in error_output:
            error_msg = f"STDERR: {error_output}"
            print(error_msg)
            return f"ERROR: {error_msg}"
            
    except subprocess.TimeoutExpired:
        timeout_msg = f"{model_name}_{gpu_count}GPU_w{workers}_b{batch_size}: TIMEOUT"
        print(timeout_msg)
        return timeout_msg
    except Exception as e:
        error_msg = f"{model_name}_{gpu_count}GPU_w{workers}_b{batch_size}: ERROR - {str(e)}"
        print(error_msg)
        return error_msg
    finally:
        # Clean up temp file
        if os.path.exists(script_path):
            os.remove(script_path)
    
    return None

def main():
    # Create results directory and file
    results_dir = Path("batch_test_results")
    results_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"batch_optimization_{timestamp}.txt"
    
    def log_both(message):
        print(message)
        with open(results_file, 'a') as f:
            f.write(message + '\n')
            f.flush()  # Force write to disk
    
    log_both("=== BATCH SIZE OPTIMIZATION TEST ===")
    log_both(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_both("Testing different batch sizes with clean memory for each test")
    log_both("")
    
    # CSV header for structured data
    csv_file = results_dir / f"batch_optimization_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Model,Resolution,GPUs,Workers,BatchSize,DataRate,ForwardTime,BackwardTime,TrainingRate,TotalThroughput,EpochTime,SamplesPerHour,Status\\n")
    
    # Test different GPU counts
    for gpu_count in [1, 2, 3]:
        log_both(f"\\n=== TESTING WITH {gpu_count} GPU(s) ===")
        
        for model_name, workers, batch_sizes, dataset_config in test_configs:
            # Determine resolution from model/dataset
            resolution = "240x240" if "240" in model_name or "240" in dataset_config else "500x500"
            
            log_both(f"\\nTesting {model_name} ({resolution}) with {gpu_count} GPU(s)")
            
            # Track if we hit memory limit
            hit_memory_limit = False
            
            for batch_size in batch_sizes:
                # Skip larger batch sizes if we already hit memory limit
                if hit_memory_limit:
                    log_both(f"  Skipping batch {batch_size} - previous OOM")
                    with open(csv_file, 'a') as f:
                        f.write(f"{model_name},{resolution},{gpu_count},{workers},{batch_size},"
                               f"0,0,0,0,0,N/A,0,SKIPPED_OOM\n")
                        f.flush()
                    continue
                
                result = run_single_test(model_name, workers, batch_size, gpu_count, dataset_config)
                
                # Log result immediately
                log_both(f"    {batch_size}: {result if result else 'No output'}")
                
                # Parse result and save to CSV
                if result and "ERROR" not in result and "TIMEOUT" not in result:
                    try:
                        # Simple parsing - look for numbers after = signs
                        data_rate = 0
                        forward_time = 0
                        backward_time = 0
                        training_rate = 0
                        total_throughput = 0
                        
                        # Extract Data rate
                        if "Data=" in result:
                            data_part = result.split("Data=")[1].split("/s")[0]
                            data_rate = float(data_part)
                        
                        # Extract Forward time
                        if "Forward=" in result:
                            forward_part = result.split("Forward=")[1].split("ms")[0]
                            forward_time = float(forward_part)
                        
                        # Extract Backward time
                        if "Backward=" in result:
                            backward_part = result.split("Backward=")[1].split("ms")[0]
                            backward_time = float(backward_part)
                        
                        # Extract Training rate
                        if "Training=" in result:
                            training_part = result.split("Training=")[1].split("/s")[0]
                            training_rate = float(training_part)
                        
                        # Extract Total throughput
                        if "Total=" in result:
                            total_part = result.split("Total=")[1].split("/s")[0]
                            total_throughput = float(total_part)
                        
                        # Calculate training speed estimates
                        if "240" in resolution:
                            dataset_size = 49000
                        else:
                            dataset_size = 34000
                        
                        if total_throughput > 0:
                            epoch_seconds = dataset_size / total_throughput
                            epoch_minutes = epoch_seconds / 60
                            if epoch_minutes < 60:
                                epoch_time = f"{epoch_minutes:.1f}min"
                            else:
                                epoch_time = f"{epoch_minutes/60:.1f}hr"
                            samples_per_hour = total_throughput * 3600
                        else:
                            epoch_time = "N/A"
                            samples_per_hour = 0
                        
                        # Write to CSV
                        with open(csv_file, 'a') as f:
                            f.write(f"{model_name},{resolution},{gpu_count},{workers},{batch_size},"
                                   f"{data_rate},{forward_time},{backward_time},"
                                   f"{training_rate},{total_throughput},{epoch_time},{samples_per_hour:.0f},SUCCESS\n")
                            f.flush()
                    except Exception as parse_error:
                        log_both(f"      Parse error: {parse_error}, result was: {result}")
                        with open(csv_file, 'a') as f:
                            f.write(f"{model_name},{resolution},{gpu_count},{workers},{batch_size},"
                                   f"0,0,0,0,0,N/A,0,PARSE_ERROR\n")
                            f.flush()
                else:
                    # Mark failed tests
                    status = "ERROR" if result and "ERROR" in result else "TIMEOUT"
                    
                    # Check if it's a memory error
                    if result and "CUDA out of memory" in result:
                        status = "OOM"
                        hit_memory_limit = True
                        log_both(f"  Memory limit reached at batch {batch_size}")
                    
                    with open(csv_file, 'a') as f:
                        f.write(f"{model_name},{resolution},{gpu_count},{workers},{batch_size},"
                               f"0,0,0,0,0,N/A,0,{status}\n")
                        f.flush()
                
                time.sleep(2)  # Brief pause between tests
    
    log_both("\\n=== TEST COMPLETED ===")
    log_both(f"Results saved to:")
    log_both(f"  Text file: {results_file}")
    log_both(f"  CSV file:  {csv_file}")
    
    # Generate summary
    log_both("\\n=== SUMMARY ANALYSIS ===")
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        log_both("\\nBest configurations by GPU count:")
        for gpu_count in [1, 2, 3]:
            gpu_data = df[(df['GPUs'] == gpu_count) & (df['Status'] == 'SUCCESS')]
            if len(gpu_data) > 0:
                best = gpu_data.loc[gpu_data['TotalThroughput'].idxmax()]
                log_both(f"  {gpu_count} GPU(s): {best['Model']} ({best['Resolution']}) w{best['Workers']} b{best['BatchSize']} "
                        f"({best['TotalThroughput']:.1f} samples/s, {best['EpochTime']} per epoch)")
        
        log_both("\\nMaximum working batch sizes:")
        for model in df['Model'].unique():
            model_data = df[(df['Model'] == model) & (df['Status'] == 'SUCCESS')]
            if len(model_data) > 0:
                max_batch = model_data['BatchSize'].max()
                resolution = model_data.iloc[0]['Resolution']
                log_both(f"  {model} ({resolution}): batch {max_batch}")
        
        log_both("\\nMemory-limited models:")
        oom_data = df[df['Status'] == 'OOM']
        if len(oom_data) > 0:
            for model in oom_data['Model'].unique():
                model_oom = oom_data[oom_data['Model'] == model]
                min_oom_batch = model_oom['BatchSize'].min()
                resolution = model_oom.iloc[0]['Resolution']
                log_both(f"  {model} ({resolution}): OOM at batch {min_oom_batch}")
        else:
            log_both("  None - all models fit in memory")
        
        log_both("\\nScaling efficiency:")
        baseline_1gpu = df[df['GPUs'] == 1]['TotalThroughput'].max()
        max_2gpu = df[df['GPUs'] == 2]['TotalThroughput'].max() 
        max_3gpu = df[df['GPUs'] == 3]['TotalThroughput'].max()
        
        if baseline_1gpu > 0:
            scaling_2gpu = max_2gpu / baseline_1gpu if max_2gpu > 0 else 0
            scaling_3gpu = max_3gpu / baseline_1gpu if max_3gpu > 0 else 0
            log_both(f"  1→2 GPUs: {scaling_2gpu:.2f}x scaling ({'Linear' if scaling_2gpu > 1.8 else 'I/O Limited'})")
            log_both(f"  1→3 GPUs: {scaling_3gpu:.2f}x scaling ({'Linear' if scaling_3gpu > 2.5 else 'I/O Limited'})")
            
    except ImportError:
        log_both("Install pandas for detailed analysis: pip install pandas")
    except Exception as e:
        log_both(f"Summary analysis failed: {str(e)}")

if __name__ == "__main__":
    import time
    main()
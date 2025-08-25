#!/usr/bin/env python
"""Test script to verify DDP MLflow logging is working correctly"""

import os
import pandas as pd
import mlflow

def check_ddp_mlflow_logging():
    """Check if DDP runs are properly logging to MLflow"""
    
    print("Checking MLflow experiments...")
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # List all experiments using search_experiments
    experiments = mlflow.search_experiments()
    print(f"\nFound {len(experiments)} experiments:")
    
    for exp in experiments:
        print(f"\n  Experiment: {exp.name} (ID: {exp.experiment_id})")
        
        # Get runs for this experiment
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  Number of runs: {len(runs)}")
        
        if len(runs) > 0:
            # Show last run details
            last_run = runs.iloc[0]
            print(f"  Last run ID: {last_run['run_id']}")
            print(f"  Last run date: {last_run['start_time']}")
            
            # Check if metrics were logged
            metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
            if metric_cols:
                print(f"  Metrics logged: {len(metric_cols)}")
                for col in metric_cols[:5]:  # Show first 5 metrics
                    print(f"    - {col}: {last_run[col]}")
            else:
                print("  WARNING: No metrics found in this run!")
                
    # Check experiment mapping file
    mapping_file = "./mlruns/experiment_mapping.json"
    if os.path.exists(mapping_file):
        print(f"\n✓ Experiment mapping file exists: {mapping_file}")
        import json
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        print(f"  Contains {len(mapping)} experiment mappings")
    else:
        print("\n✗ Experiment mapping file not found")
    
    # Check for recent DDP run directories
    print("\n\nChecking for DDP run directories...")
    ddp_base = "./model_training/working_env/ddp_runs/"
    if os.path.exists(ddp_base):
        ddp_runs = sorted([d for d in os.listdir(ddp_base) if d.startswith('ddp_run_')])
        if ddp_runs:
            print(f"Found {len(ddp_runs)} DDP run directories")
            latest_ddp = ddp_runs[-1]
            print(f"Latest: {latest_ddp}")
            
            # Check if metrics.csv exists
            metrics_csv = os.path.join(ddp_base, latest_ddp, "metrics.csv")
            if os.path.exists(metrics_csv):
                print(f"  ✓ metrics.csv exists")
                df = pd.read_csv(metrics_csv)
                print(f"  Contains {len(df)} epochs of data")
            else:
                print(f"  ✗ metrics.csv not found")
    else:
        print(f"DDP runs directory not found: {ddp_base}")

if __name__ == "__main__":
    check_ddp_mlflow_logging()
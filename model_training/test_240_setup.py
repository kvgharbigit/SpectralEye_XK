#!/usr/bin/env python3
"""
Diagnostic script to test 240x240 model setup and compare with 500x500
"""

import torch
import torch.nn as nn
from hydra import initialize, compose
from omegaconf import DictConfig
import sys
import traceback
from pathlib import Path

# Add model_training to path
sys.path.append(str(Path(__file__).parent))

def test_config_loading(config_name: str):
    """Test loading a specific config"""
    print(f"\n=== Testing config: {config_name} ===")
    try:
        with initialize(version_base=None, config_path="conf"):
            cfg = compose(config_name=config_name)
            
        print(f"✅ Config loaded successfully")
        print(f"   Model: {cfg.model.name}")
        print(f"   Target: {cfg.model.model._target_}")
        print(f"   Image size: {cfg.model.model.img_size}")
        print(f"   Spatial patch size: {cfg.model.model.spatial_patch_size}")
        print(f"   Encoder embed dim: {cfg.model.model.encoder_embed_dim}")
        print(f"   Encoder depth: {cfg.model.model.encoder_depth}")
        
        return cfg
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_model_instantiation(cfg: DictConfig):
    """Test creating the model from config"""
    print(f"\n=== Testing model instantiation ===")
    try:
        # Import the model class
        model_target = cfg.model.model._target_
        if "spectral_gpt" in model_target:
            from models.spectral_gpt.spectral_gpt import MaskedAutoencoderViT
        elif "models_mae_spectral" in model_target:
            from models.spectral_gpt.models_mae_spectral import MaskedAutoencoderViT
        else:
            raise ValueError(f"Unknown model target: {model_target}")
        
        # Create model with config parameters
        model_params = dict(cfg.model.model)
        del model_params['_target_']  # Remove hydra target
        
        print(f"   Creating model with params: {model_params}")
        model = MaskedAutoencoderViT(**model_params)
        
        print(f"✅ Model created successfully")
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        traceback.print_exc()
        return None

def test_forward_pass(model, img_size: int, batch_size: int = 2):
    """Test forward pass with synthetic data"""
    print(f"\n=== Testing forward pass (batch_size={batch_size}) ===")
    try:
        # Create synthetic input - model expects 4D: [batch, spectral, height, width]
        # Model internally adds channel dimension to make it 5D
        x = torch.randn(batch_size, 30, img_size, img_size)
        print(f"   Input shape: {x.shape}")
        
        # Test on CPU first
        model.eval()
        with torch.no_grad():
            try:
                loss, pred, mask = model(x)
                print(f"✅ CPU forward pass successful")
                print(f"   Loss: {loss.item():.4f}")
                print(f"   Prediction shape: {pred.shape}")
                print(f"   Mask shape: {mask.shape}")
                
                # Check patch calculations
                spatial_patches = img_size // model.spatial_patch_size[0]
                wavelength_patches = 30 // model.wavelength_patch_size
                total_patches = spatial_patches * spatial_patches * wavelength_patches
                print(f"   Spatial patches per dim: {spatial_patches}")
                print(f"   Total patches: {total_patches}")
                
            except Exception as e:
                print(f"❌ CPU forward pass failed: {e}")
                traceback.print_exc()
                return False
        
        # Test GPU if available
        if torch.cuda.is_available():
            print(f"   Testing on GPU...")
            device = torch.device('cuda:0')
            try:
                model_gpu = model.to(device)
                x_gpu = x.to(device)
                
                # Clear cache before test
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    loss, pred, mask = model_gpu(x_gpu)
                    print(f"✅ GPU forward pass successful")
                    print(f"   GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                    print(f"   GPU memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
                    
                # Clean up
                del model_gpu, x_gpu
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ GPU forward pass failed: {e}")
                # Print memory info for debugging
                if torch.cuda.is_available():
                    print(f"   GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                    print(f"   GPU memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 Diagnostic Script for 240x240 Model Setup")
    print("=" * 50)
    
    # Test configs to check
    configs_to_test = [
        ("config", 500),  # Original working config
        ("trial_ddp_240", 240),  # New 240x240 config
    ]
    
    results = {}
    
    for config_name, expected_img_size in configs_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING: {config_name} (expected img_size: {expected_img_size})")
        print(f"{'='*60}")
        
        # Step 1: Load config
        cfg = test_config_loading(config_name)
        if cfg is None:
            results[config_name] = "Config loading failed"
            continue
            
        # Verify image size matches expectation
        actual_img_size = cfg.model.model.img_size
        if actual_img_size != expected_img_size:
            print(f"⚠️  Image size mismatch! Expected {expected_img_size}, got {actual_img_size}")
        
        # Step 2: Create model
        model = test_model_instantiation(cfg)
        if model is None:
            results[config_name] = "Model instantiation failed"
            continue
            
        # Step 3: Test forward pass
        success = test_forward_pass(model, actual_img_size, batch_size=2)
        if success:
            results[config_name] = "✅ All tests passed"
        else:
            results[config_name] = "❌ Forward pass failed"
            
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for config_name, result in results.items():
        print(f"{config_name:20}: {result}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if "config" in results and "✅" in results["config"]:
        if "trial_ddp_240" in results and "❌" in results["trial_ddp_240"]:
            print("🔍 Original config works but 240x240 fails")
            print("   - Check model config files in conf/model/mae_*_240.yaml")
            print("   - Check dataset config in conf/dataset/dataset_240.yaml") 
            print("   - Verify patch size calculations")
        elif "trial_ddp_240" in results and "✅" in results["trial_ddp_240"]:
            print("✅ Both configs work! Issue might be in DDP setup or data loading")
            print("   - Try single GPU first: python -m model_training.train_model --config-name=trial_ddp_240")
            print("   - Check dataset path exists")
    else:
        print("❌ Original config also fails - broader issue")

if __name__ == "__main__":
    main()
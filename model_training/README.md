# Model Training Module

This module implements deep learning training pipelines for hyperspectral retinal image analysis using Masked Autoencoder Vision Transformers (MAE-ViT). It provides a comprehensive framework for training, evaluation, and visualization of models for ophthalmic image reconstruction and analysis.

## Overview

The training module implements state-of-the-art Vision Transformer architectures adapted for hyperspectral imaging data. It uses masked autoencoding as a self-supervised learning approach to learn meaningful representations from hyperspectral retinal images.

**Key Features:**
- Multiple MAE-ViT model variants (tiny to huge)
- Self-supervised learning with masked reconstruction
- Multi-GPU training support (DataParallel/DDP)
- Comprehensive experiment tracking with MLflow
- Real-time visualization during training
- Modular configuration system with Hydra

## Directory Structure

```
model_training/
├── conf/                    # Hydra configuration files
│   ├── config.yaml         # Main configuration
│   ├── accuracy/           # Accuracy metrics configs
│   ├── augmentation/       # Data augmentation configs
│   ├── dataset/            # Dataset configurations
│   ├── loss/               # Loss function configs
│   ├── metric/             # Evaluation metric configs
│   ├── model/              # Model architecture configs
│   │   ├── mae_tiny.yaml   # Tiny model configuration
│   │   ├── mae_small.yaml  # Small model configuration
│   │   ├── mae_base.yaml   # Base model configuration
│   │   ├── mae_medium.yaml # Medium model (default)
│   │   ├── mae_large.yaml  # Large model configuration
│   │   └── mae_huge.yaml   # Huge model configuration
│   ├── optimizer/          # Optimizer configs
│   ├── scheduler/          # Learning rate scheduler configs
│   └── show_prediction/    # Visualization configs
├── dataset/                # Dataset implementations
├── losses/                 # Custom loss functions
├── metrics/                # Evaluation metrics
├── mlruns/                 # MLflow experiment tracking
├── models/                 # Model architectures
│   ├── spectral_gpt/      # Spectral GPT/MAE models
│   └── spectral_autoencoder.py
├── plots/                  # Visualization utilities
├── train_val/              # Training/validation logic
├── utils/                  # Utility functions
└── working_env/            # Working directory for outputs
```

## Training Scripts

### Main Training Scripts

#### `train_model.py`
Primary training script for single-GPU training:
- Uses Hydra for configuration management
- Supports all model variants
- Integrated MLflow experiment tracking
- Real-time visualization during validation

#### `train_model_ddp.py`
Distributed Data Parallel training for multi-GPU setups:
- Supports multiple GPUs across nodes
- Synchronized batch normalization
- Gradient synchronization
- Efficient memory utilization

#### `train_model_medium.py`
Specialized training script optimized for medium-sized models:
- Pre-configured for mae_medium architecture
- Balanced performance and memory usage

## Model Architectures

### Available Models

The system implements several MAE-ViT variants optimized for hyperspectral data:

| Model | Encoder Layers | Encoder Heads | Embed Dim | Decoder Layers | Decoder Heads | Parameters |
|-------|----------------|---------------|-----------|----------------|---------------|------------|
| mae_tiny | 12 | 3 | 192 | 8 | 3 | ~5M |
| mae_small | 12 | 6 | 384 | 8 | 6 | ~22M |
| **mae_base** | **12** | **12** | **768** | **8** | **12** | **~75M** |
| mae_medium | 16 | 16 | 512 | 8 | 16 | ~86M |
| mae_large | 24 | 16 | 1024 | 8 | 16 | ~307M |
| mae_huge | 32 | 16 | 1280 | 8 | 16 | ~632M |

### Model Configuration

**Common parameters:**
- **Image size**: 500×500 pixels
- **Spectral channels**: 30 wavelengths
- **Spatial patch size**: 25×25 pixels
- **Spectral patch size**: 5 wavelengths
- **Mask ratio**: 0.8 (80% of patches masked during training)

## Configuration System

The project uses Hydra for hierarchical configuration management, enabling easy experimentation and reproducibility.

### Main Configuration (`conf/config.yaml`)

```yaml
defaults:
  - dataset: segmentation_dataset
  - model: mae_medium
  - loss: mae_spectral
  - optimizer: adamw
  - scheduler: cosine
  - show_prediction: plot_rgb

# Training hyperparameters
hparams:
  epochs: 200
  batch_size: 6
  valid_interval: 5
  accumulate_grad_batches: 1
  
# Hardware configuration  
general:
  device_ids: [0, 1, 2]  # Multi-GPU setup
  num_workers: 8
```

### Model-Specific Configurations

Each model has its own configuration file in `conf/model/`:

```yaml
# mae_medium.yaml example
_target_: models.spectral_gpt.mae_spectral.MAESpectral
img_size: 500
patch_size: [25, 5]  # [spatial, spectral]
in_chans: 30
embed_dim: 512
encoder_depth: 16
encoder_num_heads: 16
decoder_embed_dim: 256
decoder_depth: 8
decoder_num_heads: 8
mask_ratio: 0.8
```

## Usage

### Basic Training

1. **Single-GPU training with default configuration**:
   ```bash
   python train_model.py
   ```

2. **Train specific model variant**:
   ```bash
   python train_model.py model=mae_large
   ```

3. **Train MAE Base model**:
   ```bash
   python train_model.py model=mae_base
   ```

3. **Multi-GPU training**:
   ```bash
   python train_model_ddp.py
   ```

4. **Custom configuration**:
   ```bash
   python train_model.py hparams.batch_size=4 hparams.epochs=100
   ```

### Advanced Usage

#### Experiment Tracking
```bash
# Custom experiment name
python train_model.py general.experiment_name="custom_experiment"

# Custom MLflow tags
python train_model.py mlflow.tags.version="v1.0"
```

#### Model Architecture Exploration
```bash
# Tiny model for quick experiments
python train_model.py model=mae_tiny hparams.batch_size=16

# MAE Base model for ViT-Base architecture benchmarking
python train_model.py model=mae_base hparams.batch_size=6

# Large model with reduced batch size
python train_model.py model=mae_large hparams.batch_size=2
```

## Loss Functions

The system implements specialized loss functions for hyperspectral reconstruction:

### MAE Spectral Loss (`losses/mae_spectral.py`)
Combines multiple loss components:
- **Reconstruction MSE**: Standard pixel-wise reconstruction error
- **Spectral Angle Mapper (SAM)**: Preserves spectral signatures
- **Variance Loss**: Maintains spectral diversity
- **Range Penalty**: Constrains output values to valid range

### Configuration Example
```yaml
# conf/loss/mae_spectral.yaml
_target_: losses.mae_spectral.MAESpectralLoss
mse_weight: 1.0
sam_weight: 0.1
variance_weight: 0.01
range_penalty_weight: 0.001
```

## Data Pipeline

### Dataset Implementation
The system uses `SegmentationDataset` class for loading hyperspectral data:

- **Input format**: HDF5 files with shape (30, 500, 500)
- **Normalization**: Uses precomputed mean/std from data preparation
- **Augmentation**: Configurable transforms (rotation, flipping, etc.)
- **RGB support**: Optional RGB image loading for visualization

### Data Loading Configuration
```yaml
# conf/dataset/segmentation_dataset.yaml
_target_: dataset.segmentation_dataset.SegmentationDataset
csv_file: "/path/to/dataset.csv"
transform_type: "augment_basic"
normalize: true
load_rgb: true
```

## Experiment Tracking with MLflow

### MLflow Integration
- **Automatic logging**: Hyperparameters, metrics, and artifacts
- **Visualization storage**: Training images saved as PNG artifacts
- **Model checkpointing**: Best models saved with metadata
- **Experiment organization**: Hierarchical experiment structure

### Accessing Results
1. **Start MLflow UI**:
   ```bash
   cd model_training
   mlflow ui
   ```

2. **Open browser**: Navigate to `http://localhost:5000`

3. **View experiments**: Browse experiments by name (e.g., "mae_large")

4. **Inspect runs**: View metrics, parameters, and artifacts

## Visualization

### Real-time Visualization
During training, the system automatically generates visualization artifacts:

- **RGB reconstruction**: Shows model's ability to reconstruct RGB from hyperspectral
- **Original vs. predicted**: Side-by-side comparison
- **MSE per wavelength**: Reconstruction error across spectral bands
- **Spectral comparison**: Input vs. output spectra

### Visualization Configuration
```yaml
# conf/show_prediction/plot_rgb.yaml
_target_: plots.plot_bottleneck.display_rgb
nb_to_plot: 4
save_format: "png"
dpi: 150
```

## Evaluation

### Metrics
- **MSE**: Reconstruction mean squared error
- **SAM**: Spectral angle mapper
- **PSNR**: Peak signal-to-noise ratio
- **SSIM**: Structural similarity index

### Model Evaluation
Use the evaluation utilities to assess trained models:

```python
from utils.evaluate_model import evaluate_model_dataloader

# Load trained model
model = load_trained_model(checkpoint_path)

# Evaluate on test set
results = evaluate_model_dataloader(
    model=model,
    dataloader=test_loader,
    output_folder="results/"
)
```

## Hardware Requirements

### Recommended Specifications
- **GPU Memory**: 
  - mae_tiny/small: 8GB+ VRAM
  - mae_medium: 16GB+ VRAM
  - mae_large/huge: 24GB+ VRAM
- **System RAM**: 32GB+ recommended
- **Storage**: Fast SSD for data loading

### Memory Optimization
- **Gradient accumulation**: Reduce effective batch size
- **Mixed precision**: Enable automatic mixed precision training
- **Model parallelism**: Split large models across GPUs

```bash
# Enable mixed precision
python train_model.py general.use_amp=true

# Gradient accumulation
python train_model.py hparams.accumulate_grad_batches=4
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce batch size: `hparams.batch_size=2`
   - Enable gradient accumulation: `hparams.accumulate_grad_batches=4`
   - Use smaller model: `model=mae_small`

2. **Slow Training**:
   - Increase number of workers: `general.num_workers=16`
   - Use multiple GPUs: `train_model_ddp.py`
   - Enable mixed precision: `general.use_amp=true`

3. **Configuration Errors**:
   - Validate config: `python train_model.py --cfg job`
   - Check paths in dataset configuration
   - Ensure MLflow tracking URI is accessible

### Debugging

```bash
# Dry run to test configuration
python train_model.py --cfg job

# Debug mode with detailed logging
python train_model.py general.debug=true

# Single batch overfitting test
python train_model.py hparams.overfit_batches=1
```

## Best Practices

### Training Strategy
1. **Start small**: Begin with mae_tiny for quick validation
2. **Gradual scaling**: Move to larger models after confirming setup
3. **Monitor memory**: Use GPU monitoring tools during training
4. **Regular checkpointing**: Enable automatic model saving

### Experiment Organization
1. **Descriptive names**: Use meaningful experiment names
2. **Version control**: Track code changes with git
3. **Documentation**: Record experiment hypotheses and results
4. **Reproducibility**: Use fixed random seeds

### Performance Optimization
1. **Data loading**: Optimize number of workers for your system
2. **Mixed precision**: Enable for faster training on modern GPUs
3. **Distributed training**: Use multiple GPUs when available
4. **Profiling**: Monitor GPU utilization and identify bottlenecks

## Future Development

### Planned Enhancements
- **Model architectures**: Additional transformer variants
- **Loss functions**: Advanced spectral loss formulations
- **Optimization**: Further memory and speed improvements
- **Visualization**: Enhanced real-time monitoring tools

### Extension Points
- **Custom models**: Add new architectures in `models/`
- **Loss functions**: Implement domain-specific losses in `losses/`
- **Metrics**: Add evaluation metrics in `metrics/`
- **Visualizations**: Create custom plots in `plots/`
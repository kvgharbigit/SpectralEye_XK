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
- Supports multiple GPUs with proper process synchronization
- Automatic gradient averaging across all GPUs
- Memory efficient with data distribution
- Clean logging and experiment tracking (single MLflow run)
- Proper metrics aggregation from all processes

#### `train_model_medium.py`
Specialized training script optimized for medium-sized models:
- Pre-configured for mae_medium architecture
- Balanced performance and memory usage

## Model Architectures

### Available Models

The system implements several MAE-ViT variants optimized for hyperspectral data:

| Model | Encoder Layers | Encoder Heads | Embed Dim | Decoder Layers | Decoder Heads | Decoder Embed Dim | Parameters |
|-------|----------------|---------------|-----------|----------------|---------------|------------------|------------|
| mae_tiny | 4 | 4 | 128 | 2 | 2 | 64 | ~5M |
| mae_small | 8 | 8 | 256 | 4 | 4 | 128 | ~22M |
| **mae_base** | **12** | **12** | **768** | **8** | **8** | **256** | **~75M** |
| mae_medium | 16 | 16 | 512 | 8 | 8 | 256 | ~86M |
| mae_large | 12 | 12 | 960 | 8 | 8 | 256 | ~307M |
| mae_huge | 12 | 12 | 1200 | 8 | 8 | 320 | ~632M |

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
  - dataset: combined_dataset
  - model: mae_medium
  - loss: custom_loss
  - optimizer: adam
  - scheduler: none  # Default: no scheduler
  - show_prediction: plot_rgb

# Training hyperparameters
hparams:
  nb_epochs: 200
  batch_size: 5  # Per GPU in DDP mode
  lr: 0.0001
  valid_interval: 5  # Model saves and detailed plots every 5 epochs
  
# Hardware configuration  
general:
  use_amp: true  # Mixed precision enabled by default
  parallel:
    device_ids: [0, 1, 2]  # Multi-GPU setup
  num_workers: 2
```

### Learning Rate Schedulers

Available schedulers in `conf/scheduler/`:
- **none** (default): Constant learning rate
- **cosine_annealing**: Cosine decay from initial LR to 0 (SpectralGPT style)
- **reduce_on_plateau**: Reduces LR when validation loss plateaus
- **step_lr**: Step-wise LR reduction at specified intervals
- **warmup**: Linear warmup scheduler

Example with cosine annealing:
```bash
python train_model_ddp.py scheduler=cosine_annealing hparams.nb_epochs=250
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

3. **Multi-GPU DDP training** (recommended for multiple GPUs):
   ```bash
   python train_model_ddp.py model=mae_base hparams.nb_epochs=50 general.use_ddp=true
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

# DDP training for faster multi-GPU training
python train_model_ddp.py model=mae_tiny hparams.nb_epochs=30 general.use_ddp=true
```

#### Multi-GPU Training Options
```bash
# DataParallel (single process, multiple GPUs)
python train_model.py general.parallel.use_parallel=true general.parallel.device_ids=[0,1,2]

# DistributedDataParallel (multiple processes, recommended)
python train_model_ddp.py general.use_ddp=true model=mae_base

# Single GPU (specify device)
python train_model.py general.device_id=0
```

## Loss Functions

The system implements specialized loss functions for hyperspectral reconstruction:

### **Key Architectural Difference from Standard MAE**

**Important**: This implementation differs significantly from standard Masked Autoencoder approaches (like IEEE_TPAMI_SpectralGPT) in how loss is computed:

- **SpectralEye_XK**: Computes loss on **ALL patches** (both masked and unmasked)
  ```python
  recon_loss = F.mse_loss(reconstructed, original)  # Full image reconstruction
  ```

- **Standard MAE/SpectralGPT**: Computes loss **ONLY on masked patches**
  ```python
  loss = (pred - target) ** 2
  loss = (loss * mask).sum() / mask.sum()  # Only masked patches
  ```

**Implications**:
- **Training signal**: SpectralEye_XK enforces perfect reconstruction fidelity through the encoder-decoder pipeline
- **Representation learning**: Must maintain information for both masked prediction AND encoding fidelity
- **Performance**: More constrained optimization compared to standard MAE self-supervised approach

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
The system uses `CombinedDataset` class for loading hyperspectral data:

- **Input format**: HDF5 files with shape (30, 500, 500)
- **Preprocessing**: Applied during training (see below)
- **Augmentation**: Configurable transforms (rotation, flipping, etc.)
- **RGB support**: Optional RGB image loading for visualization

### Data Preprocessing During Training

The system applies several preprocessing steps to the data during training:

#### 1. Hyperspectral Data Preprocessing (`utils/preprocess_hsi.py`)
**Core transformation applied to every batch:**
```python
def preprocess_hsi(hs):
    nb_bands = hs.size(1)
    mask = hs[:, 0] < 1e-3  # Mask very low values
    mask = mask.unsqueeze(1).expand(-1, nb_bands, -1, -1).to(hs.device)
    hs[mask] = 1  # Replace masked values with 1
    hs = torch.log10(hs)  # Log10 transformation
    hs = hs + 3  # Shift by +3
    hs = hs / 3  # Scale by 1/3
    return hs
```

**Example transformation of pixel values:**

```
Normal tissue pixel:
Original: [0.15, 0.22, 0.31, 0.28, ...]
Step 1: No masking (0.15 > 1e-3)
Step 2: log10([0.15, 0.22, 0.31, 0.28, ...]) = [-0.82, -0.66, -0.51, -0.55, ...]
Step 3: Add 3: [2.18, 2.34, 2.49, 2.45, ...]
Step 4: Divide by 3: [0.73, 0.78, 0.83, 0.82, ...]

Background/invalid pixel:
Original: [0.0005, 0.0003, 0.0001, ...]  # Very low reflectance
Step 1: Mask applied → Set to 1: [1.0, 1.0, 1.0, ...]
Step 2: log10([1.0, 1.0, 1.0, ...]) = [0.0, 0.0, 0.0, ...]
Step 3: Add 3: [3.0, 3.0, 3.0, ...]
Step 4: Divide by 3: [1.0, 1.0, 1.0, ...]
```

**Purpose:**
- **Masking**: Background/invalid pixels (< 0.001) → final value 1.0
- **Log transformation**: Compresses wide dynamic range  
- **Normalization**: Maps tissue reflectance to ~[0,1], background to 1.0

#### 2. RGB Image Preprocessing (`dataset/combined_dataset.py`)
```python
rgb = imread(rgb_file).astype(np.float32) / 255.0
rgb = torch.tensor(rgb).permute(2, 0, 1)
```
- **Normalization**: RGB values normalized to [0,1] range
- **Format conversion**: HWC to CHW channel ordering

#### 3. Data Augmentation (`conf/augmentation/transform.yaml`)
Applied with 50% probability during training:
- **HorizontalFlipTransform**: Random horizontal flipping (p=0.5)
- **VerticalShiftTransform**: Vertical translation (jitter=150)
- **HorizontalShiftTransform**: Horizontal translation (jitter=150)  
- **GaussianTransform**: Gaussian noise injection (jitter=0.8)
- **RotateTransform**: Random rotation (jitter=30°)
- **ScaleTransform**: Random scaling (jitter=0.5)

#### 4. Optional Preprocessing Methods
Additional preprocessing functions available in `utils/preprocess_hsi.py`:
- **`preprocess_hsi_std_all()`**: Channel-wise normalization using precomputed statistics
- **`preprocess_hsi_xx()`**: Per-image channel-wise normalization (mean=0, std=1)
- **Random Channel Dropout**: Available in `utils/random_channel_drop.py` for regularization

### Data Loading Configuration

**Default CSV Path**: The dataset is currently configured to use:
```
F:\Foundational_model\data_500\data_all.csv
```

This path is set in `conf/dataset/combined_dataset.yaml` and can be overridden via command line:
```bash
python train_model.py dataset.csv_path="/path/to/your/dataset.csv"
```

Configuration format:
```yaml
# conf/dataset/combined_dataset.yaml
_target_: src.model_training.dataset.combined_dataset.get_dataset
csv_path: F:\Foundational_model\data_500\data_all.csv
train_ratio: 0.9
seed: 42
```

## Training Results and Output Locations

The training system stores results in multiple locations depending on the training method used:

### 1. Single GPU / DataParallel Training Output

**Location Pattern**:
```
./working_env/singlerun/{model_name}/{YYYY-MM-DD}/{HH-MM-SS}/
```

### 2. DDP Training Output

**Location Pattern**:
```
./model_training/working_env/ddp_runs/ddp_run_{YYYY-MM-DD}_{HH-MM-SS}/
```

**Single GPU Example**:
```
./working_env/singlerun/mae_medium/2025-08-23/14-30-45/
├── metrics.csv              # CSV metrics log
├── model_140.pth            # Model checkpoints (every 5 epochs)
├── model_200.pth
├── .hydra/                  # Hydra configuration snapshots
└── train_model.log          # Training logs
```

**DDP Training Example**:
```
./model_training/working_env/ddp_runs/ddp_run_2025-08-25_00-00-21/
├── metrics.csv              # Aggregated metrics from all GPUs
├── model_10.pth            # Model checkpoints (saved by rank 0 only)
├── model_20.pth
└── training.log            # Training logs
```

**Contents**:
- **CSV Metrics**: `metrics.csv` - All training metrics in CSV format
- **Model Checkpoints**: `model_{epoch}.pth` files saved at validation intervals
- **Configuration**: Complete Hydra config snapshots in `.hydra/` folder
- **Logs**: Training logs with detailed output

### 2. MLflow Experiment Tracking

**Location**: `model_training/mlruns/` directory (consistent across all training scripts)

**Structure**:
```
model_training/mlruns/
├── experiment_mapping.json  # Human-readable experiment mapping
├── 0/                       # Experiment ID directories
│   ├── meta.yaml           # Experiment metadata
│   ├── experiment_info.json # Detailed experiment info with human-readable timestamps (NEW!)
│   └── {run_id}/           # Individual run data
│       ├── metrics/        # Metrics per epoch
│       ├── params/         # All hyperparameters
│       ├── artifacts/      # Figures and visualizations
│       └── meta.yaml       # Run metadata
└── 1/                      # Another experiment
```

**Experiment Information Files**:
1. **experiment_mapping.json**: Global mapping of all experiments
2. **experiment_info.json**: Per-experiment detailed info with:
   - Human-readable creation/update timestamps
   - Full configuration details
   - Model architecture info
   - Training hyperparameters

### 3. Accessing Your Results

#### Option A: CSV Metrics (Recommended for Data Analysis)
**Location**: Each run's Hydra directory contains `metrics.csv`

**Contents**:
```csv
step,timestamp,CustomLoss train,ReconstructionMSE train,Learning Rate,CustomLoss val,ReconstructionMSE val
1,2025-08-23T14:30:45.123456,0.0234,0.0156,0.0001,0.0198,0.0134
2,2025-08-23T14:31:02.456789,0.0221,0.0149,0.0001,0.0185,0.0128
```

**Usage**:
```python
import pandas as pd
df = pd.read_csv('./working_env/singlerun/mae_medium/2025-08-23/14-30-45/metrics.csv')
print(df.head())
```

#### Option B: MLflow UI (Recommended for Interactive Exploration)
1. **Start MLflow UI**:
   ```bash
   cd model_training
   mlflow ui
   ```

2. **Open browser**: Navigate to `http://localhost:5000`

3. **View experiments**: Browse experiments by name (e.g., "mae_large")

4. **Inspect runs**: View metrics, parameters, and artifacts

#### Option C: MLflow Python API
```python
import mlflow
import pandas as pd

# Get experiment by name
experiment = mlflow.get_experiment_by_name("mae_medium")

# Get all runs from experiment
runs = mlflow.search_runs(experiment.experiment_id)

# Access metrics
metrics_df = runs[['metrics.CustomLoss train', 'metrics.ReconstructionMSE train']]
```

### 4. Metrics Logged

**Training Metrics**:
- **Train metrics**: Logged every epoch
  - Loss function value (e.g., CustomLoss, MSELoss)
  - Metric function value (e.g., ReconstructionMSE, L1Loss)
  - Learning Rate
- **Validation metrics**: Now logged **every epoch** (NEW!)
  - Lightweight validation for smooth learning curves
  - Image generation only at validation intervals (default: every 5 epochs)

**CSV Format** (Updated):
- Single row per epoch with both train and validation columns
- Validation columns populated every epoch (not just at intervals)
- Example: `step,timestamp,MSELoss train,L1Loss train,Learning Rate,MSELoss val,L1Loss val`

**Parameters Logged**:
- All Hydra configuration parameters
- Model architecture settings
- Training hyperparameters  
- Dataset configuration
- Optimizer and scheduler settings

**Artifacts Logged**:
- **Training visualizations**: Generated every 5 epochs
- **Training curves**: `training_curves.png` with 4 subplots
- **Model info**: `model_info.txt` with comprehensive training details
- **Training summary**: `training_summary.txt` with final statistics

### 5. Key Differences: Single GPU vs DDP Output

| Aspect | Single GPU/DataParallel | DDP Training |
|--------|-------------------------|--------------|
| **Output Location** | `./working_env/singlerun/` | `./working_env/ddp_runs/ddp_run_*/` |
| **MLflow Runs** | Single run | Single consolidated run (rank 0 only) |
| **Model Saving** | All processes | Rank 0 only (no duplicates) |
| **CSV Logging** | Standard location | Simplified timestamped directory |
| **Process Count** | 1 process | 1 process per GPU |
| **Metrics** | Direct | Aggregated across all GPUs |

### 6. Network Drive Backup (DDP Training)

**Automatic copying to**: `Z:\Projects\Ophthalmic neuroscience\Projects\Kayvan\SpectralEye_XK_Outputs\{run_name}\`
- **When**: Every 5 epochs and at training completion
- **Files copied**:
  - `metrics.csv` - Training metrics
  - `model_info.txt` - Model configuration details  
  - `training_curves.png` - Training/validation plots
  - `training_summary.txt` - Final statistics
  - `experiment_info.json` - Experiment metadata
- **NOT copied**: Model checkpoints (too large, kept local only)
- **Error handling**: Training continues if network drive unavailable

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
   - Use DDP to distribute memory across GPUs: `train_model_ddp.py`

2. **Slow Training**:
   - Increase number of workers: `general.num_workers=16`
   - Use DDP for multi-GPU: `python train_model_ddp.py general.use_ddp=true`
   - Enable mixed precision: `general.use_amp=true`

3. **Configuration Errors**:
   - Validate config: `python train_model.py --cfg job`
   - Check paths in dataset configuration
   - Ensure MLflow tracking URI is accessible

### DDP Training Tips
- **Batch size**: Specify per-GPU batch size (e.g., `batch_size=6` means 6 per GPU)
- **Effective batch size**: With 3 GPUs and `batch_size=6`, effective batch size is 18
- **Memory usage**: Each GPU uses similar memory to single-GPU training
- **Speed gain**: Expect ~2.5-2.8x speedup with 3 GPUs (due to communication overhead)
- **Steps per epoch**: Will be ~1/3 of single-GPU (data split across GPUs)

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
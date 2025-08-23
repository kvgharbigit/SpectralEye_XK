# Data Preparation Module

This module handles the preprocessing and preparation of hyperspectral retinal imaging data for deep learning applications. It processes raw hyperspectral data from various eye conditions and standardizes them for model training.

## Overview

The data preparation pipeline processes hyperspectral imaging data from multiple ophthalmic conditions, including:
- Control (healthy subjects)
- AMD (Age-related Macular Degeneration)
- Alzheimer's disease
- Geographic Atrophy (GA)
- Glaucoma
- Naevus and Melanomas
- Ischemia

## Directory Structure

```
data_preparation/
├── __init__.py                          # Module initialization
├── config.py                            # Central configuration file
├── copy_data.py                         # Disease-specific data processing
├── copy_data_all.py                     # Comprehensive data processing
├── create_dataframe.py                  # Dataset catalog creation
├── clean_hs_matching_rgb.py             # Data quality control
├── script_get_normalising_spectra.py    # Normalization statistics
└── data/                                # Preprocessed normalization data
    ├── mean_spectrum.npy                # Dataset mean spectrum
    └── std_spectrum.npy                 # Dataset standard deviation
```

## Key Components

### 1. Configuration (`config.py`)

Central configuration file containing all input and output paths for different data types and disease categories. Modify this file to adapt to your data structure.

Key path definitions:
- Disease-specific folders (Control, AMD, Alzheimer's, etc.)
- Reflectance data paths (Hypercolour, Optina)
- Montage data paths
- XMAS annotation paths

### 2. Data Processing Scripts

#### `copy_data.py`
Processes disease-specific hyperspectral data with the following operations:
- Reads H5 hyperspectral image files
- Downsamples wavelengths (selects 30 bands from original ~80)
- Standardizes image size to 500x500 pixels
- Organizes output by patient ID

**Key parameters:**
- Wavelength selection: indices [0:58:2, 80] → 30 bands total
- Output size: 500×500×30

#### `copy_data_all.py`
Extended processing script that handles multiple data types:
- Optina reflectance data
- Hypercolour reflectance data
- Montage data (both systems)
- Includes duplicate detection and removal
- Maintains separate validation sets

### 3. Dataset Preparation

#### `create_dataframe.py`
Creates comprehensive dataset catalogs:
- Generates pandas DataFrames with file mappings
- Creates RGB representations from hyperspectral data
- Outputs combined CSV file for training
- DataFrame columns: `ID`, `label`, `hs_file`, `rgb_file`

**RGB Generation Process:**
```python
from eitools.hs.hs_to_rgb import hs_to_rgb

def create_rgb_files(hs_files):
    for hs_file in hs_files:
        rgb_file = hs_file.parent / (hs_file.stem + '_rgb.png')
        if not rgb_file.exists():
            hs = DataHsImage(hs_file)  # Load hyperspectral cube
            rgb = hs_to_rgb(hs)        # Convert to RGB using spectral mapping
            imsave(rgb_file, np.array(rgb * 255, dtype='uint8'))
```

The RGB conversion uses the `eitools` library to map hyperspectral wavelengths to RGB colors:
- **Wavelength mapping**: 30 selected bands (450-900 nm range) converted to RGB
- **Channel mapping**: Specific bands mapped to approximate RGB channels:
  - Red: Band 12 (~615 nm)
  - Green: Band 8 (~550 nm) 
  - Blue: Band 1 (~465 nm)
- **Output format**: 8-bit PNG files (500×500×3)
- **Purpose**: Provides visual reference for hyperspectral reconstruction validation

### 4. Quality Control

#### `clean_hs_matching_rgb.py`
Ensures data integrity:
- Verifies each hyperspectral file has a corresponding RGB image
- Moves orphaned files to an "unusable" folder
- Maintains dataset consistency

### 5. Normalization

#### `script_get_normalising_spectra.py`
Calculates dataset-wide normalization parameters:
- Computes mean spectrum across all data
- Computes standard deviation spectrum
- Saves statistics for use during training

## Data Formats

### Input Format
- **File type**: HDF5 (.h5 files)
- **Structure**: Hyperspectral cubes with dimensions (wavelengths, height, width)
- **Wavelengths**: Variable (typically ~80 bands)
- **Spatial resolution**: Variable

### Output Format
- **Hyperspectral files**: HDF5 with shape (30, 500, 500)
- **RGB files**: PNG images (500×500×3)
- **Normalization**: NumPy arrays with shape (30,)
- **Dataset catalog**: CSV file with file mappings

## Usage

### Basic Workflow

1. **Configure paths** in `config.py` for your data structure

2. **Process disease-specific data**:
   ```bash
   python copy_data.py
   ```

3. **Process all data types** (optional):
   ```bash
   python copy_data_all.py
   ```

4. **Generate RGB images and create dataset catalog**:
   ```bash
   python create_dataframe.py
   ```

5. **Calculate normalization statistics**:
   ```bash
   python script_get_normalising_spectra.py
   ```

6. **Clean dataset** (remove files without RGB):
   ```bash
   python clean_hs_matching_rgb.py
   ```

### Custom Processing

To process new disease categories or data types:

1. Add new paths to `config.py`
2. Modify `copy_data.py` to include new processing logic
3. Update `create_dataframe.py` to include new data in catalog

## Dependencies

- **numpy**: Numerical operations
- **pandas**: Data management
- **matplotlib**: Visualization (debugging)
- **scikit-image**: Image processing
- **h5py**: HDF5 file handling
- **eitools**: Custom hyperspectral imaging library
  - `DataHsImage`: Read hyperspectral data
  - `hs_to_rgb`: Convert hyperspectral to RGB
  - `write_h5`: Write HDF5 files

## Output Structure

After processing, the data will be organized as:
```
output_folder/
├── patient_id_1/
│   ├── file1.h5      # Processed hyperspectral data
│   ├── file1.png     # RGB representation
│   └── ...
├── patient_id_2/
│   └── ...
└── dataset.csv       # Complete dataset catalog
```

## Notes

- Patient IDs are extracted from the first 5 characters of filenames
- The pipeline assumes Windows paths (can be adapted for other OS)
- Processing parameters (wavelength selection, image size) are currently hardcoded
- RGB generation uses the `eitools` library's hyperspectral-to-RGB conversion

## Troubleshooting

1. **Missing RGB files**: Run `clean_hs_matching_rgb.py` to identify issues
2. **Memory errors**: Process data in batches or reduce image size
3. **Path errors**: Verify all paths in `config.py` exist
4. **H5 read errors**: Ensure input files are valid HDF5 format

## Future Improvements

- Command-line arguments for flexible processing
- Parallel processing for faster execution
- Configurable wavelength selection and image sizes
- Cross-platform path handling
- Automated validation set splitting
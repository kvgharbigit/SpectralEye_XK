from eitools.image_writer import write_h5
from path import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from eitools.image_reader import DataHsImage, DataFundusImage
from skimage.transform import resize

# INPUT FOLDERS
from src.data_preparation.config import (AMD_FOLDER, CONTROL_FOLDER, GA_FOLDER, ALZHEIMER_FOLDER, GLAUCOMA_FOLDER,
                                         NEAVUS_FOLDER, ISCHAEMIA_FOLDER)

# OUTPUT FOLDERS
from src.data_preparation.config import (CONTROL_OUTPUT_FOLDER, AMD_OUTPUT_FOLDER, ALZHEIMER_OUTPUT_FOLDER,
                                         GA_OUTPUT_FOLDER, GLAUCOMA_OUTPUT_FOLDER, NEAVUS_OUTPUT_FOLDER, ISCHAEMIA_OUTPUT_FOLDER)


def get_files(folder: Path) -> list[Path]:
    files = folder.glob('*/*.h5')
    return files


def process_files(spatial_size=500):
    """Process files with configurable spatial dimensions"""
    
    for folder_in, folder_out in zip([CONTROL_FOLDER, AMD_FOLDER, ALZHEIMER_FOLDER,
                                      GA_FOLDER, GLAUCOMA_FOLDER,
                                      NEAVUS_FOLDER, ISCHAEMIA_FOLDER],
                                     [CONTROL_OUTPUT_FOLDER, AMD_OUTPUT_FOLDER, ALZHEIMER_OUTPUT_FOLDER,
                                      GA_OUTPUT_FOLDER, GLAUCOMA_OUTPUT_FOLDER,
                                      NEAVUS_OUTPUT_FOLDER, ISCHAEMIA_OUTPUT_FOLDER]):

        files = get_files(folder_in)
        print(f'Processing {folder_in}: {len(files)} files found (target size: {spatial_size}x{spatial_size})')
        for file in files:

            try:
                im = DataHsImage(file)
            except (IOError, OSError) as e:
                print(f'-------> Error while reading {file}: {e}')
                continue

            cube = im.get_cube()
            if cube is None:
                print(f'-------> Error: cube is None for file {file}')
                continue

            wl = im.wavelengths
            name = file.stem
            cube = cube[np.r_[0:58:2, 80]]
            cube = resize(cube, (30, spatial_size, spatial_size))
            wl = np.array(wl)[np.r_[0:58:2, 80]]
            save_folder = folder_out / file.parent.stem
            save_folder.makedirs_p()

            success = write_h5(cube=cube, wavelengths=wl, file_name=save_folder / f'{name}.h5')
            if not success:
                print(f'-------> Error while writing {file}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process hyperspectral images with configurable spatial dimensions')
    parser.add_argument('--spatial_size', type=int, default=500, 
                       help='Target spatial size (default: 500 for 500x500, use 240 for 240x240)')
    
    args = parser.parse_args()
    process_files(spatial_size=args.spatial_size)


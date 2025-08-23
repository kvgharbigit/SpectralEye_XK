from path import Path
import numpy as np
import pandas as pd
from eitools.hs.hs_to_rgb import hs_to_rgb
from eitools.utils.progress_bar import ProgressBar
from eitools.image_reader import DataHsImage
from skimage.io import imsave
from src.data_preparation.config import (ALZHEIMER_OUTPUT_FOLDER, AMD_OUTPUT_FOLDER, CONTROL_OUTPUT_FOLDER,
                                         GA_OUTPUT_FOLDER, GLAUCOMA_OUTPUT_FOLDER, ISCHAEMIA_OUTPUT_FOLDER,
                                         NEAVUS_OUTPUT_FOLDER, ALL_OUTPUT_FOLDER)


def create_rgb_files(hs_files):
    rgb_files = []
    for hs_file in hs_files:
        rgb_file = hs_file.parent / (hs_file.stem + '_rgb.png')
        if not rgb_file.exists():
            hs = DataHsImage(hs_file)
            rgb = hs_to_rgb(hs)
            imsave(rgb_file, np.array(rgb * 255, dtype='uint8'))
        rgb_files.append(rgb_file)
    return rgb_files


def process_optina():
    hs_files = ALL_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = ALL_OUTPUT_FOLDER.glob('*/*.png')

    reflectance_folder = Path(r'G:\Projects\Hyperspectral\Reflectance')
    unusable_reflectance_folder = Path(r'G:\Projects\Hyperspectral\Reflectance\unusable')

    for hs_file in hs_files:
        rgb_file = hs_file.parent / f'{hs_file.stem}_rgb.png'
        if not rgb_file.exists():
            hs_file.remove_p()
            # print(f'File {hs_file} does not have a corresponding RGB file.')
            reflectance_hs = reflectance_folder / hs_file.name
            if reflectance_hs.exists():
                print(f'Moved {reflectance_hs.name} to {unusable_reflectance_folder}')
                reflectance_hs.move(unusable_reflectance_folder / reflectance_hs.name)


def process_hyperolour():
    hs_files = Path(r'G:\Projects\Hyperspectral\Hypercolour_reflectance').glob('*.h5')
    rgb_folder = Path(r'G:\Projects\Hyperspectral\Hypercolour_rgb')

    pb = ProgressBar(hs_files)
    for hs_file in pb:
        rgb_file = rgb_folder / (hs_file.stem + '_rgb.png')
        if not rgb_file.exists():
            hs = DataHsImage(hs_file)
            try:
                rgb = hs_to_rgb(hs)
            except AttributeError as e:
                print(f"Error processing hyperspectral image: {e}")
                continue
            imsave(rgb_file, np.array(rgb * 255, dtype='uint8'))
    #
    # rgb_files = Path(r'G:\Projects\Hyperspectral\Hypercolour_rgb').glob('*.png')
    #
    # reflectance_folder = Path(r'G:\Projects\Hyperspectral\Hypercolour_reflectance')
    # unusable_reflectance_folder = Path(r'G:\Projects\Hyperspectral\Hypercolour_reflectance\unusable')
    #
    # for hs_file in hs_files:
    #     rgb_file = hs_file.parent / f'{hs_file.stem}_rgb.png'
    #     if not rgb_file.exists():
    #         hs_file.remove_p()
    #         # print(f'File {hs_file} does not have a corresponding RGB file.')
    #         reflectance_hs = reflectance_folder / hs_file.name
    #         if reflectance_hs.exists():
    #             print(f'Moved {reflectance_hs.name} to {unusable_reflectance_folder}')
    #             reflectance_hs.move(unusable_reflectance_folder / reflectance_hs.name)


def main():

    process_hyperolour()



if __name__ == '__main__':
    # main()

    hs_folder = Path(r'G:\Projects\Hyperspectral\Hypercolour_reflectance')
    rgb_folder = Path(r'F:\Hypercolour_rgb')
    unusable_folder = Path(r'G:\Projects\Hyperspectral\Hypercolour_reflectance\unusable')

    for hs_file in hs_folder.glob('*.h5'):
        rgb_file = rgb_folder / f'{hs_file.stem}_rgb.png'
        if not rgb_file.exists():
                print(f'Moved {hs_file.name} to {unusable_folder}')
                hs_file.move(unusable_folder)

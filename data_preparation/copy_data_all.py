from eitools.image_writer import write_h5
from path import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from eitools.image_reader import DataHsImage, DataFundusImage, get_image
from eitools.utils.progress_bar import ProgressBar
from skimage.transform import resize

# INPUT FOLDERS
from src.data_preparation.config import ALL_FOLDER, CONTROL_OUTPUT_FOLDER, AMD_OUTPUT_FOLDER, ALZHEIMER_OUTPUT_FOLDER, \
    GA_OUTPUT_FOLDER, GLAUCOMA_OUTPUT_FOLDER, NEAVUS_OUTPUT_FOLDER, ISCHAEMIA_OUTPUT_FOLDER, HYPERCOLOUR_REFLECTANCE, \
    HYPERCOLOUR_MONTAGE, HYPERCOLOUR_MONTAGE_OUTPUT, OPTINA_MONTAGE, OPTINA_MONTAGE_OUTPUT

# OUTPUT FOLDERS
from src.data_preparation.config import ALL_OUTPUT_FOLDER, HYPERCOLOUR_REFLECTANCE_OUTPUT
from src.model_training.utils.split_participants import get_unique_id_by_filename


def get_files(folder: Path) -> list[Path]:
    files = folder.glob('*.h5')
    return files


def copy_all_optina_reflectance(spatial_size=500):
    for folder_in, folder_out in zip([ALL_FOLDER], [ALL_OUTPUT_FOLDER]):

        files = get_files(folder_in)
        existing_out_files = folder_out.glob('*/*.h5')
        existing_out_files = [f.stem for f in existing_out_files]
        files = [f for f in files if f.stem not in existing_out_files]

        print(f'Processing {folder_in}: {len(files)} files found')
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

            cube = im.get_cube()
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


def copy_all_optina_montage(spatial_size=500):
    folder_in = OPTINA_MONTAGE
    folder_out = OPTINA_MONTAGE_OUTPUT

    files = get_files(folder_in)
    existing_out_files = folder_out.glob('*/*.h5')
    existing_out_files = [f.stem for f in existing_out_files]
    files = [f for f in files if f.stem not in existing_out_files]

    print(f'Processing {folder_in}: {len(files)} files found')
    pb = ProgressBar(files)
    for file in pb:
        pb.update_message(f'Processing {file.name}')
        try:
            im = get_image(file, loaded_only=True)
        except (IOError, OSError) as e:
            print(f'\r-------> Error while reading {file}: {e}')
            continue
        if im is None:
            print(f'\r-------> Error: image is None for file {file}')
            continue
        cube = im.get_cube()
        if cube is None:
            print(f'\r-------> Error: cube is None for file {file}')
            continue

        try:
            cube = im.get_cube()
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
        except Exception as e:
            print(f'\r-------> Error while processing {file}: {e}')


def copy_all_hypercolour_reflectance(spatial_size=500):
    folder_in = HYPERCOLOUR_REFLECTANCE
    folder_out = HYPERCOLOUR_REFLECTANCE_OUTPUT

    files = get_files(folder_in)
    existing_out_files = folder_out.glob('*/*.h5')
    existing_out_files = [f.stem for f in existing_out_files]
    files = [f for f in files if f.stem not in existing_out_files]

    print(f'Processing {folder_in}: {len(files)} files found')
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

        cube = im.get_cube()
        wl = im.wavelengths
        name = file.stem
        cube = resize(cube, (30, spatial_size, spatial_size))
        wl = np.array(wl)
        save_folder = folder_out / file.parent.stem
        save_folder.makedirs_p()

        success = write_h5(cube=cube, wavelengths=wl, file_name=save_folder / f'{name}.h5')
        if not success:
            print(f'-------> Error while writing {file}')


def copy_all_hypercolour_montage(spatial_size=500):
    folder_in = HYPERCOLOUR_MONTAGE
    folder_out = HYPERCOLOUR_MONTAGE_OUTPUT

    files = get_files(folder_in)
    existing_out_files = folder_out.glob('*/*.h5')
    existing_out_files = [f.stem for f in existing_out_files]
    files = [f for f in files if f.stem not in existing_out_files]

    print(f'Processing {folder_in}: {len(files)} files found')
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

        cube = im.get_cube()
        wl = im.wavelengths
        name = file.stem
        cube = resize(cube, (30, spatial_size, spatial_size))
        wl = np.array(wl)
        save_folder = folder_out / file.parent.stem
        save_folder.makedirs_p()

        success = write_h5(cube=cube, wavelengths=wl, file_name=save_folder / f'{name}.h5')
        if not success:
            print(f'-------> Error while writing {file}')


def keep_unique_validation_files():
    folder_all = ALL_OUTPUT_FOLDER
    files = folder_all.glob('*/*.h5')
    stems = [f.stem for f in files]
    folders = [CONTROL_OUTPUT_FOLDER, AMD_OUTPUT_FOLDER, ALZHEIMER_OUTPUT_FOLDER,
               GA_OUTPUT_FOLDER, GLAUCOMA_OUTPUT_FOLDER,
               NEAVUS_OUTPUT_FOLDER, ISCHAEMIA_OUTPUT_FOLDER]

    cpt = 0
    all_ids = [f.stem[:5] for f in folder_all.glob('*/*.h5')]
    for folder in folders:
        current_ids = [f.stem[:5] for f in folder.glob('*/*.h5')]

        common_ids = set(all_ids).intersection(current_ids)

        # Remove from the all_output_folder the common ids
        for common_id in common_ids:
            files = folder_all.glob(f'*/{common_id}*.h5')
            for f in files:
                f.remove()
                cpt += 1

    print(f'Found {cpt} removed files')


if __name__ == '__main__':
    # copy_all_optina_reflectance()
    # copy_all_hypercolour_reflectance()
    # keep_unique_validation_files()
    # copy_all_hypercolour_montage()
    copy_all_optina_montage()


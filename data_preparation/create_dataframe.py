from path import Path
import numpy as np
import pandas as pd
from eitools.hs.hs_to_rgb import hs_to_rgb
from eitools.image_reader import DataHsImage
from skimage.io import imsave
from src.data_preparation.config import (ALZHEIMER_OUTPUT_FOLDER, AMD_OUTPUT_FOLDER, CONTROL_OUTPUT_FOLDER,
                                         GA_OUTPUT_FOLDER, GLAUCOMA_OUTPUT_FOLDER, ISCHAEMIA_OUTPUT_FOLDER,
                                         NEAVUS_OUTPUT_FOLDER, ALL_OUTPUT_FOLDER, HYPERCOLOUR_REFLECTANCE_OUTPUT,
                                         OPTINA_MONTAGE_OUTPUT, HYPERCOLOUR_MONTAGE_OUTPUT)


def create_dataframe(hs_files, rgb_files):
    df = pd.DataFrame(columns=['ID', 'label', 'hs_file', 'rgb_file'])

    for hs_file, rgb_file in zip(hs_files, rgb_files):
        ID = hs_file.parent.stem[0:5]
        label = hs_file.parent.parent.stem
        df.loc[len(df)] = [ID, label, hs_file, rgb_file]

    return df


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


def main():

    hs_files = ALL_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_all = create_dataframe(hs_files, rgb_files)
    print(df_all)

    hs_files = ALZHEIMER_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_alzheimer = create_dataframe(hs_files, rgb_files)
    print(df_alzheimer)

    hs_files = AMD_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_amd = create_dataframe(hs_files, rgb_files)
    print(df_amd)

    hs_files = CONTROL_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_control = create_dataframe(hs_files, rgb_files)
    print(df_control)

    hs_files = GA_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_ga = create_dataframe(hs_files, rgb_files)
    print(df_ga)

    hs_files = GLAUCOMA_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_glaucoma = create_dataframe(hs_files, rgb_files)
    print(df_glaucoma)

    hs_files = ISCHAEMIA_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_ischaemia = create_dataframe(hs_files, rgb_files)
    print(df_ischaemia)

    hs_files = NEAVUS_OUTPUT_FOLDER.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_neavus = create_dataframe(hs_files, rgb_files)
    print(df_neavus)

    hs_files = HYPERCOLOUR_REFLECTANCE_OUTPUT.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_hypercolour = create_dataframe(hs_files, rgb_files)
    print(df_hypercolour)

    hs_files = OPTINA_MONTAGE_OUTPUT.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_optina_montage = create_dataframe(hs_files, rgb_files)

    hs_files = HYPERCOLOUR_MONTAGE_OUTPUT.glob('*/*.h5')
    rgb_files = create_rgb_files(hs_files)
    df_hypercolour_montage = create_dataframe(hs_files, rgb_files)


    df = pd.concat([df_all, df_alzheimer, df_amd, df_control, df_ga, df_glaucoma, df_ischaemia, df_neavus, df_hypercolour, df_optina_montage, df_hypercolour_montage], ignore_index=True)

    df.to_csv(r'F:\Foundational_model\data_500\data_all.csv', index=False)

if __name__ == '__main__':
    main()

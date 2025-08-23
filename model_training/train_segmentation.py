from path import Path
import torch
import torch.nn as nn
from models.spectral_gpt.models_mae_spectral_segmentation import mae_vit_base_patch8_256
from models.spectral_gpt.models_mae_spectral_segmentation import SegmentationHead
import numpy as np

# model_file = r'C:\Users\xhadoux\Data_projects\spectral_compression\src\model_training\working_env\singlerun\2024-12-18\17-15-38\model.pth'
model_file = r'C:\Users\xhadoux\Data_projects\spectral_compression\src\model_training\working_env\singlerun\2024-12-18\17-15-38\model.pth'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_segmentation():
    model_encoder = mae_vit_base_patch8_256()
    pretrained_dict = torch.load(model_file, weights_only=True)
    model_encoder.load_state_dict(pretrained_dict, strict=False)

    model_seg = SegmentationHead(model_encoder, num_classes=3)

    return model_seg


if __name__ == '__main__':
    model_seg = model_segmentation()
    model_seg.to(device)
    model_seg = nn.DataParallel(model_seg, device_ids=[1, 2, 3])

    data = torch.rand(1, 15, 256, 256)
    output = model_seg(data)
    print(output.shape)

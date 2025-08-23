import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCompression3D(nn.Module):
    def __init__(self, num_bands=29, out_channels=3, dropout_prob=0.0):
        """
        Args:
            num_bands (int): Number of input spectral bands.
            out_channels (int): Number of output channels for the compressed 2D output.
            dropout_prob (float): Dropout probability. Dropout layers are only added if this is greater than 0.
        """
        super(SpectralCompression3D, self).__init__()
        # size = [16, 32, 64, 128]  # Good configuration for 30 bands
        size = [16, 32, 64]  # Good configuration for 30 bands
        kernel_size = [3, 3, 3]
        self.use_dropout = dropout_prob > 0

        self.convs = nn.ModuleList()

        self.convs.append(nn.Conv3d(1, size[0], kernel_size=(kernel_size[0], 1, 1),
                                    padding=(kernel_size[0] // 2, 0, 0)))
        self.convs.append(nn.BatchNorm3d(size[0]))
        self.convs.append(nn.LeakyReLU())
        self.convs.append(nn.MaxPool3d(kernel_size=(2, 1, 1)))
        if self.use_dropout:
            self.convs.append(nn.Dropout3d(dropout_prob))

        # Subsequent Conv3D layers
        for i in range(1, len(size)):
            self.convs.append(nn.Conv3d(size[i - 1], size[i], kernel_size=(kernel_size[i-1], 1, 1),
                                        padding=(kernel_size[i-1] // 2, 0, 0)))
            self.convs.append(nn.BatchNorm3d(size[i]))
            self.convs.append(nn.LeakyReLU())
            self.convs.append(nn.MaxPool3d(kernel_size=(2, 1, 1)))
            if self.use_dropout:
                self.convs.append(nn.Dropout3d(dropout_prob))

        self.final_conv = nn.Conv3d(size[-1], out_channels, kernel_size=(num_bands // 2**len(size), 1, 1))
        self.final_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        # Add spectral dimension to input to get (B, C, S, R, C)
        for layer in self.convs:
            x = layer(x)

        x = self.final_conv(x)
        x = self.final_bn(x)
        x = F.leaky_relu(x)

        # Remove spectral dimension to return output as (B, out_channels, R, C)
        x = x.squeeze(2)  # Squeeze spectral dimension to get (B, out_channels, R, C)

        return x

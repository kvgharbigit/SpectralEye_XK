import torch
import torch.nn as nn


class SpectralAutoencoder2D(nn.Module):
    def __init__(self, spectral_layers, encoder_activation='sigmoid', decoder_activation='sigmoid',
                 use_batchnorm=False, dropout_prob=0.0):
        super(SpectralAutoencoder2D, self).__init__()

        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.layer_sizes = spectral_layers

        self.encoder = self._build(spectral_layers, encoder_activation, use_batchnorm, dropout_prob)
        self.decoder = self._build(spectral_layers[::-1], decoder_activation, use_batchnorm, dropout_prob)

    @staticmethod
    def _build(layer_sizes, activation, use_batchnorm, dropout_prob):
        """Builds the encoder layers with BatchNorm, LeakyReLU, and optional dropout."""
        layers = []
        for i in range(len(layer_sizes) - 2):
            in_channels = layer_sizes[i]
            out_channels = layer_sizes[i + 1]

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU())  # todo check where relu shoud be used

            # Apply dropout only to intermediate layers
            if dropout_prob > 0 and i < len(layer_sizes) - 2:
                layers.append(nn.Dropout(dropout_prob))

        # Last layer does not have LeakyReLU
        layers.append(nn.Conv2d(layer_sizes[-2], layer_sizes[-1], kernel_size=(1, 1)))

        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'softmax':
            layers.append(nn.Softmax(dim=1))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return compressed, reconstructed


if __name__ == '__main__':

    input_channels = 29  # Number of spectral bands in the hyperspectral image
    num_classes = 3  # Number of output classes
    rows = 256  # Number of rows in the unfolded hyperspectral image
    cols = 256  # Number of columns in the unfolded hyperspectral image
    batch_size = 3  # Number of samples in a batch
    sequence_length = rows * cols  # Length of the unfolded hyperspectral image


    model = SpectralAutoencoder2D(spectral_layers=[29, 32, 16, 3],
                                  encoder_activation='sigmoid',
                                  decoder_activation='sigmoid',
                                  use_batchnorm=True,
                                  dropout_prob=0.0)


    x = torch.randn(batch_size, input_channels, rows, cols)

    from torchinfo import summary
    summary(model, input_size=(batch_size, input_channels, rows, cols))

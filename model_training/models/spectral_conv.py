from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.init as init


class SpectralConv(nn.Module):
    def __init__(self, layer_sizes, out_activation='sigmoid', kernel_size=1, dropout_prob=0.0):
        super(SpectralConv, self).__init__()

        self.layer_sizes = layer_sizes
        self.kernel_size = kernel_size
        self.layers = nn.ModuleList()
        self.out_activation = out_activation

        # Add layers with optional batchnorm and dropout
        for i in range(1, len(layer_sizes)):
            conv_layer = nn.Conv2d(layer_sizes[i - 1], layer_sizes[i], kernel_size=1, stride=1, padding=0)
            self.layers.append(conv_layer)

            if i == len(layer_sizes) - 1:
                # No extra layers for the last convolutional layer
                continue

            self.layers.append(nn.BatchNorm2d(layer_sizes[i]))

            if dropout_prob > 0:
                self.layers.append(nn.Dropout(p=dropout_prob))

            # Initialize weights for the Conv2d layers using Xavier initialization
            init.kaiming_uniform_(conv_layer.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and i == len(self.layers) - 1:
                if self.out_activation == 'sigmoid':
                    x = torch.sigmoid(x)
                elif self.out_activation == 'softmax':
                    x = torch.softmax(x, dim=1)
            elif isinstance(layer, nn.Conv2d):
                x = torch.nn.functional.leaky_relu(x)
        return x


def main():

    input_channels = 29  # Number of spectral bands in the hyperspectral image
    num_classes = 5  # Number of output classes
    rows = 320  # Number of rows in the unfolded hyperspectral image
    cols = 320  # Number of columns in the unfolded hyperspectral image
    batch_size = 3  # Number of samples in a batch
    sequence_length = rows * cols  # Length of the unfolded hyperspectral image

    model = SpectralConv(layer_sizes=[29, 8, 6, 4, 3], out_activation='sigmoid', dropout_prob=0.0)
    x = torch.randn(batch_size, input_channels, rows, cols)

    output = model(x)

    summary(model, (batch_size, input_channels, rows, cols), device="cpu")


if __name__ == '__main__':
    main()

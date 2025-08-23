import torch
import torch.nn as nn


class ReconstructionMSE(nn.Module):
    def __init__(self):
        super(ReconstructionMSE, self).__init__()
        self.metric = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Predicted spectral values (B, W, R, C).
            targets (torch.Tensor): One-hot encoded true labels (B, W, R, C).

        Returns:
            MSE (torch.Tensor): Reconstruction metric (tensor, shape=(), scalar value).
        """
        mse = self.metric(predictions, targets)

        return mse

    def to(self, device):
        """
        Method to move the accuracy calculation to the specified device (CPU or GPU).
        Args:
            device (torch.device): The device to move the model/tensors to.
        """
        super().to(device)  # Call the parent's `to()` method to handle the device transfer
        return self  # Return self to allow chaining if necessary

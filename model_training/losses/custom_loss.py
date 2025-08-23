import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, angle_weight=0.0, variance_weight=0.0, range_weight=0.1):
        super(CustomLoss, self).__init__()

        self.reconstruction_weight = reconstruction_weight
        self.angle_weight = angle_weight
        self.variance_weight = variance_weight
        self.range_weight = range_weight


    def forward(self, reconstructed, original, latent, rgb):
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original)

        # rgb_loss = F.mse_loss(rgb_pred, rgb)
        # contrast_loss = color_moment_loss(rgb_pred, rgb)
        # latent_loss = cross_correlation_loss(latent)
        # rgb_loss = covariance_loss(latent, rgb)
        # # Unit vectors for SAM computation
        # unit_vector_C = torch.ones((1, original.size(1), 1, 1), device=original.device)  # Shape: (1, C, 1, 1)
        # unit_vector_3 = torch.ones((1, latent.size(1), 1, 1), device=latent.device)  # Shape: (1, 3, 1, 1)

        # Compute SAM angles
        # theta_original = compute_sam(original, unit_vector_C)  # Shape: (B, H, W)
        # theta_latent = compute_sam(latent, unit_vector_3)  # Shape: (B, H, W)

        # SAM Loss (Mean Squared Error between angles)
        # sam_loss = F.mse_loss(theta_latent, theta_original)

        # Contrast Enhancement Loss (Negative Variance across the batch)
        # For variance computation, flatten the spatial dimensions
        # latent_flat = latent.view(latent.size(0), latent.size(1), -1)  # Shape: (B, 3, H*W)
        # contrast_loss = -torch.mean(torch.std(latent_flat, dim=2), dim=1)  # Shape: (B, 3)
        # contrast_loss = contrast_loss.mean()  # Scalar

        # rgb_loss = uniform_loss(latent)
        # contrast_loss = decorrelation_loss(latent)
        range_loss = range_penalty_loss(latent)
        # contrast_loss = decorrelation_loss(latent)

        # Total Loss using configured weights
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.range_weight * range_loss)
        
        return total_loss


def orthogonality_loss(latent):
    B, C, H, W = latent.size()
    loss = 0.0
    for i in range(C):
        for j in range(i + 1, C):
            # Flatten spatial dimensions
            channel_i = latent[:, i, :, :].view(B, -1)  # Shape: (B, H*W)
            channel_j = latent[:, j, :, :].view(B, -1)  # Shape: (B, H*W)

            # Compute inner product
            inner_prod = torch.bmm(channel_i.unsqueeze(1), channel_j.unsqueeze(2)).squeeze()  # Shape: (B,)
            loss += torch.mean(inner_prod ** 2)

    # Average over the number of channel pairs
    num_pairs = C * (C - 1) / 2
    loss = loss / num_pairs
    return loss


def cross_correlation_loss(latent):
    B, C, H, W = latent.size()
    loss = 0.0
    for i in range(C):
        for j in range(i + 1, C):
            # Flatten spatial dimensions
            channel_i = latent[:, i, :, :].view(B, -1)  # Shape: (B, H*W)
            channel_j = latent[:, j, :, :].view(B, -1)  # Shape: (B, H*W)

            # Normalize channels
            channel_i = (channel_i - channel_i.mean(dim=1, keepdim=True)) / (channel_i.std(dim=1, keepdim=True) + 1e-8)
            channel_j = (channel_j - channel_j.mean(dim=1, keepdim=True)) / (channel_j.std(dim=1, keepdim=True) + 1e-8)

            # Compute cross-correlation
            corr = torch.mean(channel_i * channel_j, dim=1)  # Shape: (B,)
            loss += torch.mean(corr ** 2)  # Square to penalize both positive and negative correlations

    # Average over the number of channel pairs
    num_pairs = C * (C - 1) / 2
    loss = loss / num_pairs
    return loss


def gram_matrix(tensor):
    B, C, H, W = tensor.size()
    features = tensor.view(B, C, -1)  # Shape: (B, C, H*W)
    G = torch.bmm(features, features.transpose(1, 2))  # Shape: (B, C, C)
    return G / (C * H * W)


def style_loss(latent, rgb):
    G_latent = gram_matrix(latent)
    G_rgb = gram_matrix(rgb)
    loss = F.mse_loss(G_latent, G_rgb)
    return loss


def color_moment_loss(latent, rgb):
    # Compute mean
    mean_latent = latent.mean(dim=[2, 3])
    mean_rgb = rgb.mean(dim=[2, 3])

    # Compute variance
    var_latent = latent.var(dim=[2, 3])
    var_rgb = rgb.var(dim=[2, 3])

    # Compute skewness (optional)
    skew_latent = ((latent - mean_latent.unsqueeze(2).unsqueeze(3)) ** 3).mean(dim=[2, 3]) / (var_latent + 1e-8) ** 1.5
    skew_rgb = ((rgb - mean_rgb.unsqueeze(2).unsqueeze(3)) ** 3).mean(dim=[2, 3]) / (var_rgb + 1e-8) ** 1.5

    # Compute loss
    mean_loss = F.mse_loss(mean_latent, mean_rgb)
    var_loss = F.mse_loss(var_latent, var_rgb)
    skew_loss = F.mse_loss(skew_latent, skew_rgb)

    # Total color moment loss
    loss = mean_loss + var_loss + skew_loss
    return loss


def compute_covariance(tensor):
    # tensor shape: (B, C, H, W)
    B, C, H, W = tensor.size()
    tensor_flat = tensor.view(B, C, -1)  # Shape: (B, C, N), where N = H * W
    mean = tensor_flat.mean(dim=2, keepdim=True)  # Shape: (B, C, 1)
    tensor_centered = tensor_flat - mean  # Shape: (B, C, N)
    cov = torch.matmul(tensor_centered, tensor_centered.transpose(1, 2)) / (H * W - 1)  # Shape: (B, C, C)
    return cov


def covariance_loss(latent: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    cov_latent = compute_covariance(latent)
    cov_rgb = compute_covariance(rgb)
    # Compute the difference
    cov_diff: torch.Tensor = cov_latent - cov_rgb
    # Compute the Frobenius norm for each batch
    loss = torch.norm(cov_diff, dim=(1, 2))  # Shape: (B,)

    # Average over the batch
    loss = loss.mean()
    return loss


def decorrelation_loss(latent):
    """
    Encourages the latent representations to be decorrelated across channels.

    Parameters:
    - latent: Tensor of shape (B, C, H, W)

    Returns:
    - loss: Scalar tensor representing the decorrelation loss
    """
    B, C, H, W = latent.size()
    latent_flat = latent.view(B, C, -1)  # Shape: (B, C, H*W)

    # Compute the covariance matrix for each batch
    cov_matrices = []
    for b in range(B):
        latent_b = latent_flat[b]  # Shape: (C, H*W)
        mean_b = torch.mean(latent_b, dim=1, keepdim=True)  # Shape: (C, 1)
        latent_centered = latent_b - mean_b  # Shape: (C, H*W)
        cov_b = torch.matmul(latent_centered, latent_centered.t()) / (H * W - 1)  # Shape: (C, C)
        cov_matrices.append(cov_b)

    # Stack the covariance matrices and compute the mean covariance matrix
    cov_matrices = torch.stack(cov_matrices)  # Shape: (B, C, C)
    mean_cov = torch.mean(cov_matrices, dim=0)  # Shape: (C, C)

    # Compute the Frobenius norm of the off-diagonal elements
    off_diag = mean_cov - torch.diag(torch.diag(mean_cov))
    decor_loss = torch.norm(off_diag, p='fro')

    return decor_loss



def range_penalty_loss(latent):
    """
    Penalizes elements of the latent tensor that are outside the range [0, 1],
    normalized by the total number of elements.

    Parameters:
    - latent: Tensor of shape (B, C, H, W)

    Returns:
    - loss: Scalar tensor representing the normalized range penalty loss
    """
    # Compute penalties for values less than 0 and greater than 1
    penalty_below_0 = torch.relu(-latent)
    penalty_above_1 = torch.relu(latent - 1)

    # Sum the penalties
    range_penalty = penalty_below_0.sum() + penalty_above_1.sum()

    # Normalize by the total number of elements
    num_elements = latent.numel()
    normalized_range_penalty = range_penalty / num_elements

    return normalized_range_penalty


def uniform_loss(latent):
    """
    Encourages the latent representations to have the same mean and variance as a uniform distribution across each channel.

    Parameters:
    - latent: Tensor of shape (B, 3, H, W)

    Returns:
    - loss: Scalar tensor representing the uniformity loss
    """
    B, C, H, W = latent.size()

    # Mean and variance of the uniform distribution U(0,1)
    mean_uniform = 0.5
    # var_uniform = 1.0 / 12.0  # Approximately 0.0833
    var_uniform = 1.0 / 6.0  # Approximately 0.0833

    # Initialize loss
    uniformity_loss = 0.0

    for c in range(C):
        # Flatten the latent representations for each channel
        latent_flat = latent[:, c, :, :].reshape(-1)

        # Mean and variance of the latent variables for each channel
        mean_latent = torch.mean(latent_flat)
        var_latent = torch.var(latent_flat)

        # Compute the loss terms for each channel
        mean_loss = (mean_latent - mean_uniform) ** 2
        var_loss = (var_latent - var_uniform) ** 2

        # Accumulate the loss
        uniformity_loss += mean_loss + var_loss

    return uniformity_loss


def compute_sam(tensor, reference):
    """
    Computes the Spectral Angle Mapper (SAM) between each pixel vector in 'tensor' and the 'reference' vector.

    Parameters:
    - tensor: Tensor of shape (B, C, H, W)
    - reference: Tensor of shape (1, C, 1, 1)

    Returns:
    - angles: Tensor of shape (B, H, W)
    """
    # Ensure reference has shape (1, C, 1, 1)
    if reference.dim() == 1:
        reference = reference.view(1, -1, 1, 1)

    # Compute dot product along the channel dimension (C)
    dot_product = (tensor * reference).sum(dim=1)  # Shape: (B, H, W)

    # Compute norms and add small epsilon to prevent division by zero
    epsilon = 1e-8
    tensor_norm = tensor.norm(dim=1) + epsilon       # Shape: (B, H, W)
    reference_norm = reference.norm(dim=1) + epsilon  # Shape: (1, 1, 1)

    # Compute cosine similarity
    cos_theta = dot_product / (tensor_norm * reference_norm)
    # Clamp values to ensure they are within [-1, 1]
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

    # Compute angles in radians
    angles = torch.acos(cos_theta)  # Shape: (B, H, W)

    return angles
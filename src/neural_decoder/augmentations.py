import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")

class TemporalMasking(nn.Module):
    """
    SpecAugment-style time masking for neural time series.
    Randomly masks out consecutive time steps along temporal dimension.
    Args:
        max_mask_length (int): Maximum possible length of the mask.
        n_masks (int): Number of masks to apply.
        mask_value (float): Value to fill in the masked time steps.
        
    """
    def __init__(self, max_mask_length=20, n_masks=2, mask_value=0.0):
        super().__init__()
        self.max_mask_length = max_mask_length  
        self.n_masks = n_masks                  
        self.mask_value = mask_value            

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, T) or (C, T)
        Returns:
            Masked tensor with same shape
        """
        if not self.training:
            return x

        original_dim = x.dim()

        if original_dim == 2:
            x = x.unsqueeze(0) 

        B, C, T = x.shape

        for b in range(B):
            for _ in range(self.n_masks):
                mask_length = torch.randint(
                    low=1,
                    high=self.max_mask_length + 1,
                    size=(1,),
                    device=x.device,
                ).item()

                mask_length = min(mask_length, T) 


                max_start = max(1, T - mask_length)
                mask_start = torch.randint(
                    low=0,
                    high=max_start,
                    size=(1,),
                    device=x.device,
                ).item()
                x[b, :, mask_start:mask_start + mask_length] = self.mask_value

        if original_dim == 2:
            x = x.squeeze(0)  # 还原回 (C, T)

        return x
    


class LearnablePatchMasking(nn.Module):
    """
    Patch-level time masking with a learnable mask token.
    x: (B, T_patches, D)
    """
    def __init__(self, n_masks=20, mask_ratio=0.075, hidden_dim=256):
        super().__init__()
        self.n_masks = n_masks
        self.mask_ratio = mask_ratio

        # learnable mask token: shape = (1, 1, D)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x):
        """
        x: (B, T, D)
        """
        if not self.training:
            return x

        B, T, D = x.shape

        F = max(1, int(self.mask_ratio * T))

        for b in range(B):
            for _ in range(self.n_masks):
                if T - F <= 0:
                    continue
                start = torch.randint(
                    low=0, 
                    high=T - F + 1, 
                    size=(1,), 
                    device=x.device
                ).item()
                length = torch.randint(
                    low=1,
                    high=F + 1,
                    size=(1,),
                    device=x.device,
                ).item()

                end = min(T, start + length)

                x[b, start:end, :] = self.mask_token

        return x
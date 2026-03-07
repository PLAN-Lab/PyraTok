import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}.")

    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # (C,1,K,K)
    return kernel_2d


def video_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute framewise SSIM for video tensors.

    Args:
        pred:   (B, C, T, H, W)
        target: (B, C, T, H, W)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim != 5:
        raise ValueError(f"Expected 5D tensors (B,C,T,H,W), got ndim={pred.ndim}")

    bsz, channels, frames, _, _ = pred.shape
    x = pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, pred.shape[-2], pred.shape[-1])
    y = target.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, target.shape[-2], target.shape[-1])

    x = x.float()
    y = y.float()

    kernel = _gaussian_kernel_2d(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        device=x.device,
        dtype=x.dtype,
    )
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = num / (den + 1e-12)

    # Per-frame SSIM: average over channels and spatial dimensions.
    frame_ssim = ssim_map.mean(dim=(1, 2, 3)).view(bsz, frames)  # (B, T)

    if reduction == "none":
        return frame_ssim
    if reduction == "mean":
        return frame_ssim.mean()
    if reduction == "sum":
        return frame_ssim.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


class VideoSSIMLoss(nn.Module):
    """
    SSIM loss for videos in shape (B, C, T, H, W).
    Returns 1 - SSIM.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_value = video_ssim(
            pred=pred,
            target=target,
            data_range=self.data_range,
            window_size=self.window_size,
            sigma=self.sigma,
            k1=self.k1,
            k2=self.k2,
            reduction=self.reduction,
        )
        return 1.0 - ssim_value


if __name__ == "__main__":
    # Example: videos normalized to [-1, 1], so data_range=2.0.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    pred = torch.rand(2, 3, 9, 96, 160, device=device) * 2.0 - 1.0
    target = torch.rand(2, 3, 9, 96, 160, device=device) * 2.0 - 1.0

    criterion = VideoSSIMLoss(data_range=2.0, reduction="mean").to(device)
    loss = criterion(pred, target)
    print(f"SSIM loss (random videos): {loss.item():.6f}")

    identical_loss = criterion(target, target)
    print(f"SSIM loss (identical videos): {identical_loss.item():.6f}")

from .CSD.model import CSD_CLIP
from .CSD.utils import has_batchnorms, convert_state_dict
from .CSD.loss_utils import transforms_branch0
from .CSD.loss_utils import transforms_branch0
import torch
import torch.nn as nn
from typing import Callable
import torch.nn.functional as F
import torchvision.transforms as T

class DifferentiableSketch(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.blur = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.blur.weight.data[:] = 1.0 / 25.0
        self.blur.weight.requires_grad = False

    def forward(self, img):  # img: (B, C, H, W)
        if img.shape[1] == 3:
            img = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
        img = img.unsqueeze(1)

        img_blurred = self.blur(img)

        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        sketch = torch.sigmoid(5 * (grad - 0.4))
        return sketch[0, 0] # (H, W)

def load_image(image_path, image_size=256):
    transform = T.Compose([
        # T.Resize((image_size, image_size)),
        T.ToTensor(),  # Converts to range [0,1]
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # (1, 3, H, W)

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class DifferentiableAugmenter(torch.nn.Module):
    """
    Differentiable image‑to‑observation operators.
    Accepts a float tensor in N×C×H×W (C=3) and returns a tensor of the same shape.
    """

    def __init__(self):
        super().__init__()

        # --- fixed (learn‑free) kernels registered as buffers so they travel with .to(device) ---
        rgb2gray = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.register_buffer("rgb2gray", rgb2gray)

        # Sobel edge kernels (3 × 3) for "scribble"
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])
        sobel_y = sobel_x.t()
        sobel = torch.stack([sobel_x, sobel_y])         # 2×3×3
        self.register_buffer("sobel", sobel.unsqueeze(1))  # 2×1×3×3

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mode: str = "gray") -> torch.Tensor:
        """
        Args
        ----
        x    : N×3×H×W float tensor in [0, 1] or any unscaled range
        mode : "gray" | "scribble" | (... add your own)

        Returns
        -------
        Tensor with same dtype/shape & differentiable w.r.t. x
        """
        if mode == "gray":
            g = (x * self.rgb2gray).sum(1, keepdim=True)   # N×1×H×W
            return g.repeat(1, 3, 1, 1)                   # keep 3‑channel layout

        elif mode == "scribble":
            # 1. convert to gray (single channel) for stability
            g = (x * self.rgb2gray).sum(1, keepdim=True)
            # 2. Sobel gradients (padding=1 keeps spatial dims)
            gxgy = F.conv2d(g, self.sobel, padding=1)      # N×2×H×W
            grad_mag = torch.sqrt(gxgy.square().sum(1, keepdim=True) + 1e-6)
            # 3. Soft threshold → “ink on paper” look while preserving gradients
            scribble = torch.tanh(2.5 * grad_mag)
            return scribble.repeat(1, 3, 1, 1)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def visualize(img_tensor: torch.Tensor, title: str = "") -> None:
        """
        Quick inline visualizer (Jupyter‑friendly).
        Accepts N×C×H×W or C×H×W; clamps to [0, 1] for display.
        """
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        grid = make_grid(img_tensor.cpu().clamp(0, 1), nrow=min(4, img_tensor.size(0)))
        plt.figure(figsize=(4 * grid.size(2) / grid.size(1), 4))
        if title:
            plt.title(title)
        plt.axis("off")
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()
        
def setup_csd(device: str = "cpu") -> tuple[nn.Module, Callable]:
    """Sets up the CSD model.

    Args:
        device: The device to load the model onto.

    Returns:
        The initialized CSD model and preprocess function.
    """
    model = CSD_CLIP("vit_large", "default")
    model_path = "model/CSD-ViT-L/pytorch_model.bin"
    if has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    state_dict = convert_state_dict(checkpoint["model_state_dict"])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded checkpoint with msg {msg}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    preprocess = transforms_branch0
    return model.to(device), preprocess



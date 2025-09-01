from typing import Callable, Tuple

import torch


def create_gradient_mask(shape: Tuple, feather: int, device="cpu") -> torch.Tensor:
    """
    Create a gradient mask for smooth blending of tiles.

    Args:
        shape (tuple): Shape of the mask (batch, channels, height, width)
        feather (int): Width of the feathered edge

    Returns:
        torch.Tensor: Gradient mask
    """
    mask = torch.ones(shape).to(device)
    _, _, h, w = shape
    for feather_step in range(feather):
        factor = (feather_step + 1) / feather
        mask[:, :, feather_step, :] *= factor
        mask[:, :, h - 1 - feather_step, :] *= factor
        mask[:, :, :, feather_step] *= factor
        mask[:, :, :, w - 1 - feather_step] *= factor
    return mask


def tiled_upscale(
    samples: torch.Tensor,
    function: Callable,
    scale: int,
    tile_width: int = 512,
    tile_height: int = 512,
    overlap: int = 8,
) -> torch.Tensor:
    """
    Apply a scaling function to image samples in a tiled manner.

    Args:
        samples (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        function (Callable): The scaling function to apply to each tile
        scale (int): Factor by which to upscale the image
        tile_width (int): Width of each tile
        tile_height (int): Height of each tile
        overlap (int): Overlap between tiles

    Returns:
        torch.Tensor: Upscaled and processed output tensor
    """
    _batch, _channels, height, width = samples.shape
    out_height, out_width = round(height * scale), round(width * scale)
    output_device = samples.device

    # Initialize output tensors
    output = torch.empty((1, 3, out_height, out_width), device=output_device)
    out = torch.zeros((1, 3, out_height, out_width), device=output_device)
    out_div = torch.zeros_like(output)

    # Process the image in tiles
    for y in range(0, height, tile_height - overlap):
        for x in range(0, width, tile_width - overlap):
            # Ensure we don't go out of bounds
            x_end = min(x + tile_width, width)
            y_end = min(y + tile_height, height)
            x = max(0, x_end - tile_width)
            y = max(0, y_end - tile_height)

            # Extract and process the tile
            tile = samples[:, :, y:y_end, x:x_end]
            processed_tile = function(tile).to(output_device)

            # Calculate the position in the output tensor
            out_y, out_x = round(y * scale), round(x * scale)
            out_h, out_w = processed_tile.shape[2:]

            # Create a feathered mask for smooth blending
            mask = create_gradient_mask(processed_tile.shape, overlap * scale, device=output_device)

            # Add the processed tile to the output
            out[:, :, out_y : out_y + out_h, out_x : out_x + out_w] += processed_tile * mask
            out_div[:, :, out_y : out_y + out_h, out_x : out_x + out_w] += mask

    # Normalize the output
    output = out / out_div

    return output

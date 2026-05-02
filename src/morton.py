from torch import Tensor

def spread_bits(x: Tensor) -> Tensor:
    """Spread the lower 21 bits of x so that they occupy bit positions 0,3,6, ... ,60."""
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x

def morton_code(coords: Tensor) -> Tensor:
    """Convert (x,y,z) integer coords into 64-bit Morton codes (position along Z-order curve)."""
    return ((spread_bits(coords[..., 0]) << 2) | (spread_bits(coords[..., 1]) << 1) |  spread_bits(coords[..., 2]))

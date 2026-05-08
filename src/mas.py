import torch
from torch import Tensor

from morton import morton_code
import acceleration
device = acceleration.get_device()


def build_schwarz_domains(
    positions: Tensor,
    domain_size: int = 32
):
    # morton code and sorting stuff
    node_count = positions.shape[0]

    bb_min = positions.min(dim=0).values
    bb_max = positions.max(dim=0).values

    extent = (bb_max - bb_min).clamp(min=1e-8)
    grid_res = (1 << 20) - 1

    normalized = (positions - bb_min) / extent
    grid_coords = (normalized * grid_res).long()

    codes = morton_code(grid_coords)
    _, sort_idx = torch.sort(codes)

    # partitioning stuff
    num_domains = (node_count + domain_size - 1) // domain_size
    pad = num_domains * domain_size - node_count

    if pad > 0:
        sort_idx = torch.cat([
            sort_idx,
            sort_idx.new_full((pad,), -1)
        ])

    domains = sort_idx.view(num_domains, domain_size)
    return domains

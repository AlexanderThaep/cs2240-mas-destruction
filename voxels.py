import torch
from dataclasses import dataclass

@dataclass
class ConnectedVoxels:
    pos: torch.Tensor    # (N,3) int or float grid coords
    links: torch.Tensor  # (N,6)

    @staticmethod
    def from_positions(pos):
        vox = ConnectedVoxels(pos=pos, links=None)
        vox.build_links()
        return vox

    def build_links(self):
        pos = self.pos

        # ensure integer grid coords
        coords = pos.int()

        # hashset: (x,y,z) -> index
        hashmap = {
            (int(x), int(y), int(z)): i
            for i, (x, y, z) in enumerate(coords)
        }

        N = coords.shape[0]
        links = torch.full((N, 6), -1, dtype=torch.int32)

        offsets = [
            (-1,0,0),
            ( 1,0,0),
            (0,-1,0),
            (0, 1,0),
            (0,0,-1),
            (0,0, 1),
        ]

        for i, (x,y,z) in enumerate(coords):
            for d, (dx,dy,dz) in enumerate(offsets):
                neighbor = (int(x+dx), int(y+dy), int(z+dz))
                j = hashmap.get(neighbor, -1)
                links[i, d] = j

        self.links = links
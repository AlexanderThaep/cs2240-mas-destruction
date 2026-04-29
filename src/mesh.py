import torch
import importlib.util
from dataclasses import dataclass

@dataclass
class Mesh:
    path:         str
    colormap:     torch.Tensor
    voxelmap:     torch.Tensor

    def info(self):
        print(self.path)
        print(self.voxelmap.shape)
        print(self.colormap.shape)

    @staticmethod
    def from_py(path: str):
        spec = importlib.util.spec_from_file_location(
            "mesh_module",
            path
        )

        if not spec: return None
        if not spec.loader: return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        lookup = module.lookup
        palette = module.pallette

        # -----------------------------
        # voxel tensor
        # [x,y,z,color_idx]
        # -----------------------------
        voxelmap = torch.tensor(
            [
                [
                    v["x"],
                    v["y"],
                    v["z"],
                    v["color"]
                ]
                for v in lookup
            ],
            dtype=torch.int32
        )

        # -----------------------------
        # palette tensor
        # normalize RGB -> [0,1]
        # -----------------------------
        colormap = torch.tensor(
            [
                [
                    c["red"] / 255.0,
                    c["green"] / 255.0,
                    c["blue"] / 255.0,
                ]
                for c in palette
            ],
            dtype=torch.float32
        )

        return Mesh(
            path=path,
            colormap=colormap,
            voxelmap=voxelmap
        )
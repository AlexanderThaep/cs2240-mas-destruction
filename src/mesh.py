import torch
import importlib.util
import struct
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

    def translate(self, offset):
        self.voxelmap[:, :3] += torch.Tensor(offset).to(self.voxelmap.dtype)
        return self

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

    @staticmethod
    def from_vox(path: str) -> "Mesh":
        """Load a MagicaVoxel .vox file. Z-up source coords are remapped to Y-up."""
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"VOX ":
            raise ValueError(f"Not a MagicaVoxel file: {path}")

        # Header is "VOX " + version (8 bytes), then a single MAIN chunk whose
        # children carry SIZE / XYZI / RGBA. We only need XYZI (and RGBA if present).
        pos = 8
        main_content_size = struct.unpack("<I", data[pos+4:pos+8])[0]
        main_children_size = struct.unpack("<I", data[pos+8:pos+12])[0]
        pos += 12 + main_content_size
        end = pos + main_children_size

        xyzi_chunks = []
        palette = None
        while pos < end:
            chunk_id = data[pos:pos+4]
            cs = struct.unpack("<I", data[pos+4:pos+8])[0]
            chs = struct.unpack("<I", data[pos+8:pos+12])[0]
            content = data[pos+12:pos+12+cs]
            if chunk_id == b"XYZI":
                n = struct.unpack("<I", content[:4])[0]
                arr = torch.frombuffer(bytearray(content[4:4+4*n]), dtype=torch.uint8).reshape(n, 4)
                xyzi_chunks.append(arr)
            elif chunk_id == b"RGBA":
                palette = torch.frombuffer(bytearray(content), dtype=torch.uint8).reshape(256, 4)
            pos += 12 + cs + chs

        if not xyzi_chunks:
            raise ValueError(f"No XYZI chunks in {path}")

        raw = torch.cat(xyzi_chunks, dim=0).long()  # (N,4) [x, y_vox, z_vox, color]
        # MagicaVoxel is Z-up; this codebase is Y-up (see voxels.NODE_OFFSETS).
        voxelmap = torch.stack([raw[:, 0], raw[:, 2], raw[:, 1], raw[:, 3]], dim=1).to(torch.int32)

        if palette is None:
            colormap = torch.full((256, 3), 0.6, dtype=torch.float32)
        else:
            colormap = palette[:, :3].float() / 255.0

        return Mesh(path=path, colormap=colormap, voxelmap=voxelmap)
from typing import Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from klay.core import ModuleCategory, register
from klay.layers._base import _BaseLayer


@register(
    "SphericalHarmonicEdgeAttrsNoShift",
    inputs=["pos", "edge_index"],
    outputs=["edge_vec", "edge_lengths", "edge_sh"],
    category=ModuleCategory.EMBEDDING,
)
@compile_mode("script")
class SphericalHarmonicEdgeAttrsNoShift(_BaseLayer, torch.nn.Module):
    def __init__(self, lmax: int, normalization: Optional[str] = "component"):
        super().__init__()
        irreps_edge_sh = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_in = lmax
        self.irreps_out = irreps_edge_sh
        edge_sh_normalize = normalization is not None
        self.sh = o3.SphericalHarmonics(
            irreps_edge_sh, edge_sh_normalize, normalization
        )

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor):
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_lengths = torch.linalg.norm(edge_vec, dim=1)
        edge_sh = self.sh(edge_vec)
        return edge_vec, edge_lengths, edge_sh

    @classmethod
    def from_config(cls, *, lmax: int = 1, normalization: Optional[str] = "component"):
        return cls(lmax=lmax, normalization=normalization)

import math
from typing import Tuple, Union
import jetnet
import torch
from torch import nn

EPS = 1e-16


class EMDLoss(nn.Module):
    """EMDLoss Wrapper for jetnet.losses.EMDLoss"""

    def __init__(self, *args, **kwargs):
        """EMDLoss Wrapper for jetnet.losses.EMDLoss"""
        super(EMDLoss, self).__init__()
        self.emd_loss = jetnet.losses.EMDLoss(*args, **kwargs)

    def forward(
        self,
        p_recons: torch.Tensor,
        p_target: torch.Tensor
    ) -> Union[torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor]]:
        if (p_recons.shape[-1] != 4) and (p_recons.shape[-1] != 3):
            raise ValueError(f"p_recons must be a 3- or 4-vector. Got: {p_recons.shape=}")
        if (p_target.shape[-1] != 4) and (p_target.shape[-1] != 3):
            raise ValueError(
                f"p_target must be a 3- or 4-vector. Got: {p_target.shape=}")

        return self.emd_loss(
            self.__get_p3_rel(p_recons),
            self.__get_p3_rel(p_target),
            return_flows=False
        ).sum(0)  # sum over batch

    def __get_p3_polar(self, p: torch.Tensor) -> torch.Tensor:
        """(E, px, py, pz) or (px, py, pz) -> (eta, phi, pt)"""
        if p.shape[-1] == 4:  # 4-vectors
            p0, px, py, pz = p.unbind(-1)
        else:
            px, py, pz = p.unbind(-1)
        pt = torch.sqrt(px ** 2 + py ** 2 + EPS)
        phi = torch.atan2(py + EPS, px + EPS)
        eta = torch.asinh(pz / (pt + EPS))
        return torch.stack([eta, phi, pt], dim=-1)

    def __get_p3_rel(self, p: torch.Tensor) -> torch.Tensor:
        """(E, px, py, pz) -> (eta_rel, phi_rel, pt_rel)"""
        eta, pt, phi = self.__get_p3_polar(p).unbind(dim=-1)
        jet_eta, jet_pt, jet_phi = (self.__get_p3_polar(
            p.sum(dim=-2)).unsqueeze(-2)).unbind(-1)
        # eta_rel
        eta_rel = eta - jet_eta
        # phi_rel
        phi_rel = phi - jet_phi
        phi_rel = (phi_rel + math.pi) % (2 * math.pi) - math.pi
        # pt_rel
        pt_rel = pt / (jet_pt + EPS)
        return torch.stack([eta_rel, phi_rel, pt_rel], dim=-1)

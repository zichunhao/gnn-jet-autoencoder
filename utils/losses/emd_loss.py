import math
from typing import Tuple, Union
import jetnet
import torch
from torch import nn

EPS = 1e-16


class EMDLoss(nn.Module):
    """EMDLoss Wrapper for jetnet.losses.EMDLoss"""

    def __init__(
        self,
        polar_coord: bool = False,
        abs_coord: bool = True,
        *args, **kwargs
    ):
        """
        EMDLoss Wrapper for jetnet.losses.EMDLoss
        
        :param polar_coord: Use polar coordinates for EMD loss.
        :param abs_coord: Use absolute coordinates for EMD loss.
        - (polar_coord, abs_coord) = (True, True): (pt, eta, phi)
        - (polar_coord, abs_coord) = (True, False): (pt_rel, eta_rel, phi_rel)
        - (polar_coord, abs_coord) = (False, True): (px, py, pz)
        - (polar_coord, abs_coord) = (False, False): (px_rel, py_rel, pz_rel)
        """
        super(EMDLoss, self).__init__()
        self.emd_loss = jetnet.losses.EMDLoss(*args, **kwargs)
        self.polar_coord = polar_coord
        self.abs_coord = abs_coord

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
            self.__get_p3_polarrel(p_recons),
            self.__get_p3_polarrel(p_target),
            return_flows=False
        ).sum(0)  # sum over batch

    def __get_p3_polar_from_cartesian(self, p: torch.Tensor) -> torch.Tensor:
        """(E, px, py, pz) or (px, py, pz) -> (pt, eta, phi)"""
        if p.shape[-1] == 4:  # 4-vectors
            _, px, py, pz = p.unbind(-1)
        else:
            px, py, pz = p.unbind(-1)
        
        pt = torch.sqrt(px ** 2 + py ** 2 + EPS)
        phi = torch.atan2(py + EPS, px + EPS)
        eta = torch.asinh(pz / (pt + EPS))
        
        return torch.stack([pt, eta, phi], dim=-1)
    
    def __get_p4_cartesian_from_polar(self, p: torch.Tensor) -> torch.Tensor:
        """
        (E, pt, eta, phi) or (pt, eta, phi) -> (E, px, py, pz)
        assuming massless particles
        """
        if p.shape[-1] == 4:
            # 4-vectors
            p0, pt, eta, phi = p.unbind(-1)
        else:
            pt, eta, phi = p.unbind(-1)
            p0 = pt * torch.cosh(eta)  # assuming massless particles
        
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        
        return torch.stack([p0, px, py, pz], dim=-1)

    def __get_p3_polarrel(self, p: torch.Tensor) -> torch.Tensor:
        """p -> (eta_rel, phi_rel, pt_rel)
        
        - (polar_coord, abs_coord) = (True, True): p = (pt, eta, phi)
        - (polar_coord, abs_coord) = (True, False): p = (pt_rel, eta_rel, phi_rel)
        - (polar_coord, abs_coord) = (False, True): p = (px, py, pz)
        - (polar_coord, abs_coord) = (False, False): p = (px_rel, py_rel, pz_rel)
        """
        if self.abs_coord:
            # absolute/ coordinates
            if self.polar_coord:
                # (pt, eta, phi)
                if p.shape[-1] == 4:
                    _, pt, eta, phi = p.unbind(-1)
                else:  # p.shape[-1] == 3
                    pt, eta, phi = p.unbind(-1)
                
                # get jet momenta components in polar coordinates
                p4_cartesian = self.__get_p4_cartesian_from_polar(p)
                jet_p_cartesian = p4_cartesian.sum(-2)
                jet_p_polar = self.__get_p3_polar_from_cartesian(jet_p_cartesian)
                jet_pt, jet_eta, jet_phi = jet_p_polar.unbind(-1)
            
            else:
                # (px, py, pz)
                pt, eta, phi = self.__get_p3_polar_from_cartesian(p).unbind(dim=-1)
                # get jet momenta components in polar coordinates
                jet_p_cartesian = p.sum(dim=-2, keepdim=True)
                jet_p_polar = self.__get_p3_polar_from_cartesian(jet_p_cartesian)
                jet_pt, jet_eta, jet_phi = jet_p_polar.unbind(dim=-1)
            
            # eta_rel
            eta_rel = eta - jet_eta
            # phi_rel
            phi_rel = phi - jet_phi
            phi_rel = (phi_rel + math.pi) % (2 * math.pi) - math.pi
            # pt_rel
            pt_rel = pt / (jet_pt + EPS)
            
        else:
            # relative coordinates
            if self.polar_coord:
                # (pt_rel, eta_rel, phi_rel)
                if p.shape[-1] == 4:
                    _, pt_rel, eta_rel, phi_rel = p.unbind(-1)
                else:  # p.shape[-1] == 3
                    pt_rel, eta_rel, phi_rel = p.unbind(-1)
            
            else:
                # (px_rel, py_rel, pz_rel)
                p_polarrel = self.__get_p3_polar_from_cartesian(p)
                pt_rel, eta_rel, phi_rel = p_polarrel.unbind(-1)
    
        return torch.stack([eta_rel, phi_rel, pt_rel], dim=-1)

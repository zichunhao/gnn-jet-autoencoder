from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from models import Encoder, Decoder
from utils.const import DEFAULT_DEVICE

EPS = 1e-12

class PermutationTest:
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DEVICE
    ):
        self.encoder = encoder.to(device=device, dtype=dtype)
        self.decoder = decoder.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        
    def __call__(
        self, 
        x: Union[torch.Tensor, DataLoader],
        verbose: bool = False,
        save_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, DataLoader):
            # dataloader multiple batches
            inv_dev = []
            eqv_dev = []
            perm_inv = []
            perm_eqv = []
            dataloader = x
            for x in dataloader:
                x = x.to(device=self.device, dtype=self.dtype)
                inv, p_inv = self.invariance_dev(x)
                eqv, p_eqv = self.equivariance_dev(x)
                inv_dev.append(inv.detach().cpu())
                eqv_dev.append(eqv.detach().cpu())
                perm_inv.append(p_inv.detach().cpu())
                perm_eqv.append(p_eqv.detach().cpu())
            inv_dev = torch.cat(inv_dev, dim=0)
            eqv_dev = torch.cat(eqv_dev, dim=0)
            perm_inv = torch.cat(perm_inv, dim=0)
            perm_eqv = torch.cat(perm_eqv, dim=0)
        elif isinstance(x, torch.Tensor):
            x = x.to(device=self.device, dtype=self.dtype)
            # a single batch
            inv_dev, perm_inv = self.invariance_dev(x)
            eqv_dev, perm_eqv = self.equivariance_dev(x)
            inv_dev = inv_dev.detach().cpu()
            eqv_dev = eqv_dev.detach().cpu()
            perm_inv = perm_inv.detach().cpu()
            perm_eqv = perm_eqv.detach().cpu()
        else:
            raise TypeError(
                "x must be a DataLoader or a Tensor. "
                f"Found: {type(x)}"
            )
        
        if save_dir is not None:
            path_inv = Path(save_dir) / "invariance.pt"
            path_eqv = Path(save_dir) / "equivariance.pt"
        else:
            path_inv = None
            path_eqv = None
        return {
            'invariance': get_dev_summary(inv_dev, perm=perm_inv, verbose=verbose, save_path=path_inv),
            'equivariance': get_dev_summary(eqv_dev, perm=perm_eqv, verbose=verbose, save_path=path_eqv)
        }
    
    def invariance_dev(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check if :math:`\mathrm{NN}(P(x)) = \mathrm{NN}(x)`, 
        where :math:`P` is a random permutation.
        """
        # NN(x)
        y = get_model_output(x, self.encoder, self.decoder)
        
        # NN(P(x))
        x_perm, perm = particle_perm_rand(x)
        y_perm = get_model_output(x_perm, self.encoder, self.decoder)
        
        # NN(P(x)) = NN(x)?
        return dev(output=y_perm, target=y), perm

    def equivariance_dev(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check if :math:`\mathrm{NN}(P(x)) = P(\mathrm{NN}(x))`, 
        where :math:`P` is a random permutation.
        """
        # NN(x)
        y = get_model_output(x, self.encoder, self.decoder)
        
        # NN(P(x))
        x_perm, perm = particle_perm_rand(x)
        y_perm = get_model_output(x_perm, self.encoder, self.decoder)
        
        # NN(P(x)) = P(NN(x))?
        return dev(output=y_perm, target=apply_perm(perm, y)), perm

def dev(
    output: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """Deviation of output from target."""
    return (output - target).abs() / (target.abs() + EPS)

def apply_perm(
    perm: torch.Tensor, 
    x: torch.Tensor
) -> torch.Tensor:
    """Apply a permutation to a tensor."""
    return torch.stack([x[idx, p] for (idx, p) in enumerate(perm)])

def particle_perm_rand(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly permute the particle features within a jet.
    
    :param x: particle features of shape (batch_size, num_particles, vec_dims)
    :return: (x_perm, perm) where x_perm is the permuted particle features,
    and perm is the permutation.
    :rtype: (torch.Tensor, torch.Tensor)
    """ 
    # generate a random permutation
    batch_size, num_particles, _ = x.shape
    perm = 1 * torch.arange(num_particles).expand(batch_size, -1)
    for idx in range(batch_size):
        perm[idx, :] = torch.randperm(num_particles)
    
    # apply the permutation
    return apply_perm(perm, x), perm

def get_model_output(
    x: torch.Tensor, 
    encoder: Encoder, 
    decoder: Decoder
) -> torch.Tensor:
    """Get the model output for a given input."""
    return decoder(encoder(x)).detach()

def get_dev_summary(
    dev: torch.Tensor, 
    perm: torch.Tensor,
    verbose: bool = False,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Union[torch.Tensor, float]]:
    
    summary = {
        'mean': dev.mean().item(),
        'median': dev.median().item(),
        'max': dev.max().item(),
        'min': dev.min().item(),
        'std': dev.std().item(),
    }
    
    if verbose:
        # include the permutation and deviation
        perm['values'] = dev
        if perm is not None:
            summary['perm'] = perm
            
    if save_path is not None:
        # save the summary to a file
        torch.save(summary, save_path)
    return summary
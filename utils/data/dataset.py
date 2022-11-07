from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset


class JetMomentaDataset(Dataset):
    """
    PyTorch dataset for GNN autoencoder.
    """

    def __init__(
        self, 
        data: Union[torch.Tensor, np.ndarray], 
        vec_dims: int = 3,
        polar_coord: bool = True,
    ):
        # input checks
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            pass
        else:
            raise TypeError(
                'Expected data to be a numpy.ndarray or a torch.Tensor'
                f'Found: {type(data)}'
            )

        if vec_dims not in (3, 4):
            raise ValueError(f"vec_dims must be 3 or 4. Found: {vec_dims}")

        if data.shape[-1] == 3:
            if vec_dims == 4:
                # expand to 4-vector assuming massless particles
                if polar_coord:
                    pt, eta, phi = data.unbind(-1)
                    p0 = pt * torch.cosh(eta)
                else:
                    p0 = torch.norm(data, dim=-1, keepdim=True)
                # (|p|, px, py, pz) or (|p|, pt, eta, phi)
                self.data = torch.cat([p0, data], dim=-1)
            else:
                self.data = data
        elif data.shape[-1] == 4:
            if vec_dims == 3:
                self.data = data[..., 1:]
            else:
                self.data = data
            

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

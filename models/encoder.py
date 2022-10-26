from typing import List, Union
import torch
import torch.nn as nn
import logging

from .graphnet import GraphNet

from utils.const import DEFAULT_DEVICE, DEFAULT_DTYPE
from .const import LOCAL_MIX, GLOBAL_MIX

class Encoder(nn.Module):
    def __init__(
        self, 
        num_nodes: int, 
        input_node_size: int, 
        latent_node_size: int, 
        node_sizes: List[List[int]], 
        edge_sizes: List[List[int]],
        num_mps: int, 
        dropout: float, 
        alphas: List[int],
        batch_norm: bool = False, 
        latent_map: str = 'global mix', 
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        """GNN encoder built on `GraphNet`

        :param num_nodes: Number of nodes in the graph.
        :type num_nodes: int
        :param input_node_size: Size/dimension of the input feature vectors. 
        :type input_node_size: int
        :param latent_node_size: Size/dimension of the latent feature vectors.
        If `encoder_level` is 'global mix', this is the size of the global latent vector.
        If `encoder_level` is 'local mix', this is the size of the local latent vectors, 
        so that the total size is `latent_node_size * num_nodes`.
        :type latent_node_size: int
        :param node_sizes: List of sizes/dimensions of the node feature vectors 
        in each massage passing step.
        :type node_sizes: List[List[int]]
        :param edge_sizes: List of sizes/dimensions of the edge feature vectors 
        in each massage passing step.
        :type edge_sizes: List[List[int]]
        :param num_mps: Number of message passing steps.
        :type num_mps: int
        :param dropout: Dropout rate.
        :type dropout: float
        :param alphas: Alpha value for the leaky relu layer for edge features 
        in each iteration of message passing.
        :type alphas: List[int]
        :param batch_norm: Whether to use batch normalization 
        in the edge and node features, defaults to False
        :type batch_norm: bool, optional
        :param latent_map: Choice of mapping to latent space, defaults to 'node'. 
        If `'mean'`, the mean is taken across the node features in the graph.
        If `'local'` or `'node'`, linear layers are applied per node. 
        If `'global'` or `'graph'`, a single linear layer is applied to the graph. 
        :type latent_map: str, optional
        :param device: Device of the model, defaults to None. 
        If None, use gpu if cuda is available and otherwise cpu.
        :type device: torch.device, optional
        :param dtype: Dtype of the model, defaults to None.
        If None, use torch.float64.
        :type dtype: torch.dtype, optional
        """
        if device is None:
            device = DEFAULT_DEVICE
        if dtype is None:
            dtype = DEFAULT_DTYPE

        super(Encoder, self).__init__()

        self.num_nodes = num_nodes
        self.input_node_size = input_node_size
        self.latent_node_size = latent_node_size
        self.node_sizes = node_sizes
        self.edge_sizes = edge_sizes
        self.num_mps = num_mps
        self.latent_map = latent_map
        if self.latent_map.lower().replace(' ', '_') in LOCAL_MIX:
            # (30 x 3 -> 30 x dim)
            self.latent_space_size = latent_node_size * num_nodes
        else:
            self.latent_space_size = latent_node_size

        self.device = device
        self.dtype = dtype

        # layers
        if self.latent_map.lower() in LOCAL_MIX:
            encoder_out_size = self.node_sizes[-1][-1]
        else:
            encoder_out_size = self.latent_node_size
        
        self.encoder = GraphNet(
            num_nodes=self.num_nodes, 
            input_node_size=input_node_size,
            output_node_size=encoder_out_size,
            node_sizes=self.node_sizes,
            edge_sizes=self.edge_sizes, 
            num_mps=self.num_mps,
            dropout=dropout, 
            alphas=alphas, 
            batch_norm=batch_norm,
            dtype=self.dtype, 
            device=self.device
        ).to(self.device).to(self.dtype)
        
        if self.latent_map.lower().replace(' ', '_') in GLOBAL_MIX:
            self.mix_layer = nn.Linear(
                self.latent_node_size*self.num_nodes,
                self.latent_node_size, 
                bias=False
            ).to(self.device).to(self.dtype)
        elif self.latent_map.lower().replace(' ', '_') in LOCAL_MIX:
            self.mix_layer = nn.Linear(
                encoder_out_size, latent_node_size
            ).to(self.device).to(self.dtype)
        else:
            pass

        num_params = sum(
            p.nelement() 
            for p in self.parameters() 
            if p.requires_grad
        )

        logging.info(
            f"Encoder initialized. Number of parameters: {num_params}"
        )

    def forward(
        self, 
        x: torch.Tensor, 
        metric: str = 'euclidean',
    ) -> torch.Tensor:
        bs = x.shape[0]
        x = x.to(self.device).to(self.dtype)
        x = self.encoder(x, metric=metric)
        x = self.__aggregate(x, bs, self.latent_map)
        return x

    def __aggregate(self, x, bs, latent_map):
        latent_map = self.latent_map.replace(' ', '_').lower()
        # aggregation to latent space
        if latent_map == 'mean':
            # (bs, n, graph_dim) -> (bs, latent_dim)
            x = torch.mean(x, dim=-2)
        elif latent_map == 'max':
            # (bs, n, graph_dim) -> (bs, latent_dim)
            x = torch.amax(x, dim=-2)
        elif latent_map == 'min':
            # (bs, n, graph_dim) -> (bs, latent_dim)
            x = torch.amin(x, dim=-2)
        elif latent_map.replace(' ', '_') in GLOBAL_MIX:
            # (bs, n, graph_dim) -> (bs, n*graph_dim) -> (bs, latent_dim)
            x = self.mix_layer(x.view(bs, -1))
        elif latent_map.replace(' ', '_') in LOCAL_MIX:
            # (bs, n, graph_dim) -> (bs, n, latent_dim)
            x = self.mix_layer(x).view(bs, -1)
        else:
            # default to "mean"
            logging.warning(f"Unknown latent map {self.latent_map} in Encoder. Using mean.")
            x = self.aggregate(x, bs, latent_map='mean')
        
        logging.debug(f"Encoder output shape: {x.shape}")
        return x

    def l1_norm(self):
        """L1 norm of the model parameters."""
        return sum(p.abs().sum() for p in self.parameters())


    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(p.pow(2).sum() for p in self.parameters())
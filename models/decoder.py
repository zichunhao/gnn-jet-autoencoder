from typing import List, Union
import torch
import torch.nn as nn
import logging

from .graphnet import GraphNet
from .const import LOCAL_MIX
from utils.const import DEFAULT_DEVICE, DEFAULT_DTYPE

class Decoder(nn.Module):
    def __init__(
        self, 
        num_nodes: int, 
        latent_node_size: int, 
        output_node_size: int, 
        node_sizes: List[List[int]], 
        edge_sizes: List[List[int]],
        num_mps: int, 
        alphas: List[int], 
        dropout: float = 0.0,
        batch_norm: int = False, 
        latent_map: str = 'mix', 
        normalize_output: bool = False,
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        """GNN decoder built on `GraphNet`

        :param num_nodes: Number of nodes in the decoder.
        :type num_nodes: int
        :param latent_node_size: Size/dimension of the latent feature vectors.
        If the latent map is 'local mix', this is the size of the latent feature vectors per node.
        :type latent_node_size: int
        :param output_node_size: Size/dimension of the output/reconstructed node feature vectors.
        :type output_node_size: int
        :param node_sizes: List of sizes/dimensions of the node feature vectors 
        in each massage passing step.
        :type node_sizes: Union[int, List[int], List[int]]
        :param edge_sizes: List of sizes/dimensions of the edge feature vectors 
        in each massage passing step.
        :type edge_sizes: Union[int, List[int], List[int]]
        :param num_mps: Number of message passing steps.
        :type num_mps: int
        :param alphas: Alpha value for the leaky relu layer for edge features 
        in each iteration of message passing.
        :type alphas: Union[int, List[int]]
        :param dropout: Dropout rate, defaults to 0.0.
        :type dropout: float
        :param batch_norm: Whether to use batch normalization 
        in the edge and node features., defaults to False
        :type batch_norm: int, optional
        :param latent_map: Choice of mapping to latent space, defaults to 'node'. 
        If `'mean'`, the mean is taken across the node features in the graph.
        If `'local'` or `'node'`, linear layers are applied per node. 
        If `'global'` or `'graph'`, a single linear layer is applied to the graph. 
        :type latent_map: str, optional
        :param normalize_output: Whether to normalize output, defaults to False
        :type normalize_output: bool, optional
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

        super(Decoder, self).__init__()

        self.num_nodes = num_nodes
        self.latent_map = latent_map
        self.latent_node_size = latent_node_size
        self.output_node_size = output_node_size
        self.node_sizes = node_sizes
        self.edge_sizes = edge_sizes
        self.num_mps = num_mps
        self.normalize_output = normalize_output

        self.device = device
        self.dtype = dtype

        # layers
        if self.latent_map.lower().replace(' ', '_') in LOCAL_MIX:
            # node-wise aggregation layer
            self.linear = nn.Linear(
                latent_node_size,
                self.latent_node_size
            ).to(self.device).to(self.dtype)
        else:
            self.linear = nn.Linear(
                latent_node_size,
                self.num_nodes*self.latent_node_size
            ).to(self.device).to(self.dtype)
            

        self.decoder = GraphNet(
            num_nodes=self.num_nodes, 
            input_node_size=self.latent_node_size,
            output_node_size=self.output_node_size, 
            node_sizes=self.node_sizes,
            edge_sizes=self.edge_sizes, 
            num_mps=self.num_mps,
            alphas=alphas, 
            dropout=dropout,
            batch_norm=batch_norm,
            dtype=self.dtype, 
            device=self.device
        ).to(self.device).to(self.dtype)

    def forward(
        self, 
        x: torch.Tensor, 
        metric='euclidean'
    ) -> torch.Tensor:
        x = self.__prepare_input(x)
        # graph net
        x = self.decoder(x, metric=metric)
        if self.normalize_output:
            x = torch.tanh(x)
        return x

    def __prepare_input(self, x):
        """Prepare input for the graph decoder."""
        x = x.to(self.device).to(self.dtype)
        if self.latent_map.lower().replace(' ', '_') in LOCAL_MIX:
            x = x.view(-1, self.num_nodes, self.latent_node_size)
            x = self.linear(x)  # map to input node size
        else:
            x = self.linear(x)  # map to input node size
            x = x.view(-1, self.num_nodes, self.latent_node_size)
        return x
    
    def l1_norm(self):
        """L1 norm of the model parameters."""
        return sum(p.abs().sum() for p in self.parameters())


    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(p.pow(2).sum() for p in self.parameters())
    
    @property
    def num_learnable_params(self):
        return sum(p.nelement()for p in self.parameters() if p.requires_grad)

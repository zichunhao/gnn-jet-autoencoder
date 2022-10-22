from typing import Any, Callable, Iterable, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import types

from utils.const import DEFAULT_DEVICE, DEFAULT_DTYPE, EPS

class GraphNet(nn.Module):

    def __init__(
        self, 
        num_nodes: int, 
        input_node_size: int, 
        output_node_size: int, 
        node_sizes: List[List[int]], 
        edge_sizes: List[List[int]],
        num_mps: int, 
        dropout: float = 0.1, 
        alphas: List[int] = 0.1,
        batch_norm: bool = False, 
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        """A fully connected message-passing standard graph neural network
        with the distance as edge features.

        :param num_nodes: Number of nodes in the graph.
        :type num_nodes: int
        :param input_node_size: Size/dimension of input node feature vectors.
        :type input_node_size: int
        :param output_node_size: Size/dimension of output node feature vectors.
        :type output_node_size: int
        :param node_sizes: Sizes/dimensions of hidden node 
        in each layer in each iteration of message passing.
        :type node_sizes: List[List[int]]
        :param edge_sizes: Sizes/dimensions of edges networks 
        in each layer  in each iteration of message passing.
        :type edge_sizes: List[List[int]]
        :param num_mps: Number of message passing steps.
        :type num_mps: int
        :param dropout: Dropout rate for edge conv, defaults to 0.1
        :type dropout: float, optional
        :param alphas: Alpha value for the leaky relu layer for edge features 
        in each iteration of message passing., defaults to 0.1
        :type alphas: List[int], optional
        :param batch_norm: Whether to use batch normalization, defaults to False
        :type batch_norm: bool, optional
        :param device: Device of the model, defaults to None. 
        If None, use gpu if cuda is available and otherwise cpu.
        :type device: torch.device, optional
        :param dtype: Dtype of the model, defaults to None.
        If None, use torch.float64.
        :type dtype: torch.dtype, optional
        """
        
        node_sizes = _adjust_var_list(node_sizes, num_mps)
        edge_sizes = _adjust_var_list(edge_sizes, num_mps)
        alphas = _adjust_var_list(alphas, num_mps)

        if device is None:
            device = DEFAULT_DEVICE
        if dtype is None:
            dtype = DEFAULT_DTYPE

        super(GraphNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.eps = EPS

        # Node networks
        self.num_nodes = num_nodes  # Number of nodes in graph
        self.input_node_size = input_node_size  # Dimension of input node features
        self.node_sizes = node_sizes  # List of dimensions of hidden node features
        self.node_net = nn.ModuleList()
        self.output_node_size = output_node_size  # Dimension of output node features
        self.output_layer = nn.Linear(self.node_sizes[-1][-1], self.output_node_size)

        # Edge networks
        self.edge_sizes = edge_sizes  # Output sizes in edge networks
        # mij = xi ⊕ xj ⊕ d(xi, xj)
        self.input_edge_sizes = [2 * s[0] + 1 for s in self.node_sizes]
        self.edge_net = nn.ModuleList()

        self.num_mps = num_mps  # Number of message passing
        self.batch_norm = batch_norm  # Use batch normalization (bool)
        if self.batch_norm:
            self.bn_node = nn.ModuleList()
            self.bn_edge = nn.ModuleList()

        self.alphas = alphas  # For leaky relu layer for edge features
        self.dropout_p = dropout  # Dropout layer for edge features

        for i in range(self.num_mps):
            # Edge layers
            edge_layers = _create_dnn(
                layer_sizes=self.edge_sizes[i], 
                input_size=self.input_edge_sizes[i]
            )
            self.edge_net.append(edge_layers)
            if self.batch_norm:
                bn_edge_i = nn.ModuleList()
                for j in range(len(edge_layers)):
                    bn_edge_i.append(nn.BatchNorm1d(self.edge_sizes[i][j]))
                self.bn_edge.append(bn_edge_i)

            # Node layers
            node_layers = _create_dnn(layer_sizes=self.node_sizes[i])
            node_layers.insert(0, nn.Linear(
                self.edge_sizes[i][-1] + self.node_sizes[i][0], self.node_sizes[i][0]
            ))
            if i+1 < self.num_mps:
                node_layers.append(nn.Linear(node_sizes[i][-1], self.node_sizes[i+1][0]))
            else:
                node_layers.append(nn.Linear(node_sizes[i][-1], self.output_node_size))
            self.node_net.append(node_layers)
            if self.batch_norm:
                bn_node_i = nn.ModuleList()
                for j in range(len(self.node_net[i])):
                    bn_node_i.append(nn.BatchNorm1d(self.node_net[i][j].out_features))
                self.bn_node.append(bn_node_i)

        self.to(device=self.device, dtype=self.dtype)

    def forward(
        self, 
        x: torch.Tensor, 
        metric: str = 'euclidean'
    ) -> torch.Tensor:
        """Forward pass of the model.

        :param x: Input node features.
        :type x: torch.Tensor
        :param metric: Metric to compute the distance, defaults to 'cartesian'. 
        Choice: ('Minkowskian', 'euclidean'). 
        'Minkowskian' is only available for 4-vectors. 
        :type metric: str, optional
        :return: Output node features.
        :rtype: torch.Tensor
        """
        self.metric = metric.lower()
        batch_size = x.shape[0]
        x = x.to(device=self.device, dtype=self.dtype)
        x = F.pad(x, (0, self.node_sizes[0][0] - self.input_node_size, 0, 0, 0, 0))

        for i in range(self.num_mps):
            metric = _get_metric_func(self.metric if x.shape[-1] == 4 else 'cartesian')
            A = self._getA(x, batch_size, self.input_edge_sizes[i], hidden_node_size=self.node_sizes[i][0], metric=metric)
            A = self._edge_conv(A, i)
            x = self._aggregate(x, A, i, batch_size)
            x = x.view(batch_size, self.num_nodes, -1)

        x = x.view(batch_size, self.num_nodes, self.output_node_size)
        return x

    def _getA(
        self, 
        x: torch.Tensor, 
        batch_size: int, 
        input_edge_size: int, 
        hidden_node_size: int, 
        metric: Callable
    ) -> torch.Tensor:
        """Get Adjacency matrix A

        :param x: node embeddings
        :type x: torch.Tensor
        :param batch_size: Batch size
        :type batch_size: int
        :param input_edge_size: Size of input edge features
        :type input_edge_size: int
        :param hidden_node_size: Size of hidden node features
        :type hidden_node_size: int
        :param metric: Metric for computing distances between nodes
        :type metric: Callable
        :return: Adjacency matrix that stores distances among nodes. 
        Shape: (batch_size * self.num_nodes * self.num_nodes, input_edge_size)
        :rtype: torch.Tensor
        """
        x1 = x.repeat(1, 1, self.num_nodes).view(batch_size, self.num_nodes*self.num_nodes, hidden_node_size)
        x2 = x.repeat(1, self.num_nodes, 1)  # 1*(self.num_nodes)*1 tensor with repeated x along axis=1
        dists = metric(x2 - x1 + self.eps)
        A = torch.cat((x1, x2, dists), 2).view(batch_size*self.num_nodes*self.num_nodes, input_edge_size)
        return A

    def _concat(
        self, 
        A: torch.Tensor, 
        x: torch.Tensor, 
        batch_size: int, 
        edge_size: int, 
        node_size: int
    ) -> torch.Tensor:
        """Concatenate node features and edge features

        :param A: Adjacency matrix that stores distances among nodes.
        :type A: torch.Tensor
        :param x: Node features.
        :type x: torch.Tensor
        :param batch_size: Batch size.
        :type batch_size: int
        :param edge_size: Edge feature size.
        :type edge_size: int
        :param node_size: Node feature size.
        :type node_size: int
        :return: Concatenated edge and node features.
        :rtype: torch.Tensor
        """        
        A = A.view(batch_size, self.num_nodes, self.num_nodes, edge_size)
        A = torch.sum(A, 2)
        x = torch.cat((A, x), 2)
        x = x.view(batch_size * self.num_nodes, edge_size + node_size)
        return x

    def _dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Dropout layer

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """        
        dropout = nn.Dropout(p=self.dropout_p)
        return dropout(x)

    def _aggregate(
        self, 
        x: torch.Tensor, 
        A: torch.Tensor, 
        i: int, 
        batch_size: int
    ) -> torch.Tensor:
        """Aggregate node features with edge features

        :param x: Node embeddings.
        :type x: torch.Tensor
        :param A: (Convoluted) adjacency matrix that stores distances among nodes.
        :type A: torch.Tensor
        :param i: Layer index.
        :type i: int
        :param batch_size: Batch size.
        :type batch_size: int
        :return: Aggregated node embeddings.
        :rtype: torch.Tensor
        """
        x = self._concat(A, x, batch_size, self.edge_sizes[i][-1], self.node_sizes[i][0])
        for j in range(len(self.node_net[i])):
            x = self.node_net[i][j](x)
            x = F.leaky_relu(x, negative_slope=self.alphas[i])
            if self.batch_norm:
                x = self.bn_node[i][j](x)
            x = self._dropout(x)
        return x

    def _edge_conv(
        self, 
        A: torch.Tensor, 
        i: int
    ) -> torch.Tensor:
        """Edge convolution at layer i

        :param A: Adjacency matrix that stores distances among nodes.
        :type A: torch.Tensor
        :param i: Layer index.
        :type i: int
        :return: Convolved adjacency matrix.
        :rtype: torch.Tensor
        """        
        for j in range(len(self.edge_net[i])):
            A = self.edge_net[i][j](A)
            A = F.leaky_relu(A, negative_slope=self.alphas[i])
            if self.batch_norm:
                A = self.bn_edge[i][j](A)
            A = self._dropout(A)
        return A


def _create_dnn(
    layer_sizes: List[int], 
    input_size: int = -1
) -> nn.ModuleList:
    dnn = nn.ModuleList()
    if input_size >= 0:
        sizes = layer_sizes.copy()
        sizes.insert(0, input_size)
        for i in range(len(layer_sizes)):
            dnn.append(nn.Linear(sizes[i], sizes[i+1]))
    else:
        for i in range(len(layer_sizes) - 1):
            dnn.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    return dnn


def _adjust_var_list(data, num):
    try:
        if len(data) < num:
            data = data + [data[-1]] * (num - len(data))
            return data
    except TypeError:  # Non interable
        data = [data] * num
    return data[:num]

# def _adjust_var_list(
#     data: Union[List[List[Any]], List[Any], Any], 
#     num: int
# ) -> List[List[Any]]:
#     if isinstance(data, list):
#         if isinstance(data[0], list):
#             # list of list
#             if len(data) < num:
#                 data = data + [data[-1]] * (num - len(data))
#             else:
#                 data = data[:num]
#             return data
#         else:
#             # list -> list of list
#             return [data] * num


def _get_metric_func(metric: Union[Callable, str]) -> Callable:
    if isinstance(metric, Callable):
        return metric
    if metric.lower() in ('cartesian', 'euclidean'):
        return lambda x: torch.sum(x**2, -1).unsqueeze(-1)
    if metric.lower() == 'minkowskian':
        return lambda x: (2 * torch.pow(x[..., 0], 2) - torch.sum(torch.pow(x, 2), dim=-1)).unsqueeze(-1)
    else:
        logging.warning(f"Metric ({metric}) for adjacency matrix is not implemented. Use 'cartesian' instead.")
        return _get_metric_func('cartesian')

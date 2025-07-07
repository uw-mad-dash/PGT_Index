import math
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn.conv import MessagePassing


class DConv(MessagePassing):
    r"""An implementation of the Diffusion Convolution Layer.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`).

    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(DConv, self).__init__(aggr="add", flow="source_to_target")
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(2, K, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.__reset_parameters()

    def __reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        adj_mat = to_dense_adj(edge_index, edge_attr=edge_weight)
        adj_mat = adj_mat.reshape(adj_mat.size(1), adj_mat.size(2))
        deg_out = torch.matmul(
            adj_mat, torch.ones(size=(adj_mat.size(0), 1)).to(X.device)
        )
        deg_out = deg_out.flatten()
        deg_in = torch.matmul(
            torch.ones(size=(1, adj_mat.size(0))).to(X.device), adj_mat
        )
        deg_in = deg_in.flatten()

        deg_out_inv = torch.reciprocal(deg_out)
        deg_in_inv = torch.reciprocal(deg_in)
        row, col = edge_index
        norm_out = deg_out_inv[row]
        norm_in = deg_in_inv[row]

        reverse_edge_index = adj_mat.transpose(0, 1)
        reverse_edge_index, vv = dense_to_sparse(reverse_edge_index)

        Tx_0 = X
        Tx_1 = X
        H = torch.matmul(Tx_0, (self.weight[0])[0]) + torch.matmul(
            Tx_0, (self.weight[1])[0]
        )

        if self.weight.size(1) > 1:
            Tx_1_o = self.propagate(edge_index, x=X, norm=norm_out, size=None)
            Tx_1_i = self.propagate(reverse_edge_index, x=X, norm=norm_in, size=None)
            H = (
                H
                + torch.matmul(Tx_1_o, (self.weight[0])[1])
                + torch.matmul(Tx_1_i, (self.weight[1])[1])
            )

        for k in range(2, self.weight.size(1)):
            Tx_2_o = self.propagate(edge_index, x=Tx_1_o, norm=norm_out, size=None)
            Tx_2_o = 2.0 * Tx_2_o - Tx_0
            Tx_2_i = self.propagate(
                reverse_edge_index, x=Tx_1_i, norm=norm_in, size=None
            )
            Tx_2_i = 2.0 * Tx_2_i - Tx_0
            H = (
                H
                + torch.matmul(Tx_2_o, (self.weight[0])[k])
                + torch.matmul(Tx_2_i, (self.weight[1])[k])
            )
            Tx_0, Tx_1_o, Tx_1_i = Tx_1, Tx_2_o, Tx_2_i

        if self.bias is not None:
            H += self.bias

        return H


class DCRNN(torch.nn.Module):
    r"""An implementation of the Diffusion Convolutional Gated Recurrent Unit.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`)

    """

    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super(DCRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([X, H], dim=1)
        Z = self.conv_x_z(Z, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([X, H], dim=1)
        R = self.conv_x_r(R, edge_index, edge_weight)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([X, H * R], dim=1)
        H_tilde = self.conv_x_h(H_tilde, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.
            * **H** (PyTorch Float Tensor, optional) - Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class BatchedDConv(MessagePassing):
    r"""Implementation of the  Diffusion Convolution Layer that enables batching and seq-to-seq prediction
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`).

    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(BatchedDConv, self).__init__(aggr="add", flow="source_to_target")
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(2, K, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.__reset_parameters()

    def __reset_parameters(self):
        # torch.nn.init.ones_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor,
        cached_idx = False
    ) -> torch.FloatTensor:
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """

        if not cached_idx:
            row, col = edge_index
            deg_out = torch.zeros(X.size(0), device=X.device).scatter_add_(0, row, edge_weight)
            deg_in = torch.zeros(X.size(0), device=X.device).scatter_add_(0, col, edge_weight)
            
            deg_out_inv = torch.reciprocal(deg_out)
            deg_in_inv = torch.reciprocal(deg_in)
            row, col = edge_index
            self._cached_norm_out = deg_out_inv[row]
            self._cached_norm_in = deg_in_inv[row]

            reverse_edge_index = torch.stack([col, row], dim=0)
            sort_idx = reverse_edge_index[0] * X.size(0) + reverse_edge_index[1]
            self._cached_reverse_edge_index = reverse_edge_index  = reverse_edge_index[:, sort_idx.argsort()]


        Tx_0 = X
        Tx_1 = X
        H = torch.matmul(Tx_0, (self.weight[0])[0]) + torch.matmul(
            Tx_0, (self.weight[1])[0]
        )
    
        if self.weight.size(1) > 1:
            Tx_1_o = self.propagate(edge_index, x=X, norm=self._cached_norm_out, size=None)
            Tx_1_i = self.propagate(self._cached_reverse_edge_index, x=X, norm=self._cached_norm_in, size=None)
            H = (
                H
                + torch.matmul(Tx_1_o, (self.weight[0])[1])
                + torch.matmul(Tx_1_i, (self.weight[1])[1])
            )

        for k in range(2, self.weight.size(1)):
            Tx_2_o = self.propagate(edge_index, x=Tx_1_o, norm=self._cached_norm_out, size=None)
            Tx_2_o = 2.0 * Tx_2_o - Tx_0
            Tx_2_i = self.propagate(
                self._cached_reverse_edge_index, x=Tx_1_i, norm=self._cached_norm_in, size=None
            )
            Tx_2_i = 2.0 * Tx_2_i - Tx_0
            H = (
                H
                + torch.matmul(Tx_2_o, (self.weight[0])[k])
                + torch.matmul(Tx_2_i, (self.weight[1])[k])
            )
            Tx_0, Tx_1_o, Tx_1_i = Tx_1, Tx_2_o, Tx_2_i

        if self.bias is not None:
            H += self.bias
       
        return H


class BatchedDCRNN(torch.nn.Module):
    """
    Implementation of the  Diffusion Convolutional Recurrent Neural Network that enables batching and seq-to-seq prediction.
    The input data is expected to be of shape `(batch_size, seq_length, num_nodes, num_features)`.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`)

    """
    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super(BatchedDCRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias

        self._create_parameters_and_layers()

        self._cached_batch_size = None
        self._cached_edge_index = None
        self._cached_edge_weight = None


        self._cached_expanded_edge_index = None
        self._cached_expanded_edge_weight = None

        self._cached_idx = False

    def _replicate_edge_index(self, edge_index, batch_size, num_nodes):
        edge_index = edge_index.clone()  # clone once to avoid modifying original
        repeated = []
        for i in range(batch_size):
            offset = i * num_nodes
            repeated.append(edge_index + offset)
        return torch.cat(repeated, dim=1)


    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = BatchedDConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = BatchedDConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = BatchedDConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, cached):
        
        Z = torch.cat([X, H], dim=1)
        Z = self.conv_x_z(Z, edge_index, edge_weight, cached_idx = cached)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, cached):
        R = torch.cat([X, H], dim=1)
        R = self.conv_x_r(R, edge_index, edge_weight, cached_idx = cached)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, cached):
        H_tilde = torch.cat([X, H * R], dim=1)
        H_tilde = self.conv_x_h(H_tilde, edge_index, edge_weight, cached_idx = cached)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X, edge_index, edge_weight):
        """
        Forward pass for DCRNN with batching and sequence support.

        Args:
            X: Input tensor of shape (batch_size, seq_length, num_nodes, num_features)
            edge_index: Edge index for the graph
            edge_weight: Edge weights for the graph

        Returns:
            Output tensor of shape (batch_size, seq_length, num_nodes, out_channels)
        """

        
        batch_size, seq_length, num_nodes, num_features = X.size()
        hidden_state = torch.zeros(batch_size, num_nodes, self.out_channels).to(X.device)

        if self._cached_edge_index == None or self._cached_batch_size != batch_size \
        or not torch.equal(self._cached_edge_index, edge_index) or not torch.equal(self._cached_edge_weight, edge_weight):
 
            # cache for future comparision to check freshness 
            self._cached_batch_size = batch_size
            self._cached_edge_index = edge_index
            self._cached_edge_weight = edge_weight

            # cache
            self._cached_expanded_edge_index = self._replicate_edge_index(edge_index, batch_size, num_nodes)
            self._cached_expanded_edge_weight = edge_weight.repeat(batch_size)

            self._cached_idx = False
        else:
            self._cached_idx = True

        outputs = []
        for t in range(seq_length):
            x_t = X[:, t, :, :].reshape(batch_size * num_nodes, num_features)
            
            H = hidden_state.reshape(batch_size * num_nodes, self.out_channels)
            Z = self._calculate_update_gate(x_t, self._cached_expanded_edge_index, self._cached_expanded_edge_weight, H, self._cached_idx)
            R = self._calculate_reset_gate(x_t, self._cached_expanded_edge_index, self._cached_expanded_edge_weight, H, self._cached_idx)
            H_tilde = self._calculate_candidate_state(x_t, self._cached_expanded_edge_index, self._cached_expanded_edge_weight, H, R, self._cached_idx)
            H = self._calculate_hidden_state(Z, H, H_tilde)
          
            hidden_state = H.reshape(batch_size, num_nodes, self.out_channels)
            outputs.append(hidden_state)

        return torch.stack(outputs, dim=1)
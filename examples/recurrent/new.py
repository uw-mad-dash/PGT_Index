import torch
from torch_geometric.data import Batch
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.utils import to_dense_batch


class PyGDiffusionConvolution(DCRNN):
    def __init__(self, node_features, hidden_size, k):
        super(PyGDiffusionConvolution, self).__init__(node_features, hidden_size, k)

    def multi_step_prediction(self, x, edge_index, edge_weight=None, steps=1, batch=None):
        """
        Performs multi-step forecasting.
        Args:
            x (torch.FloatTensor): Node features (batch_size, num_nodes, features).
            edge_index (torch.LongTensor): Edge index of the graph(s).
            edge_weight (torch.FloatTensor, optional): Edge weights of the graph(s).
            steps (int): Number of prediction steps ahead.

        Returns:
            torch.FloatTensor: Forecasted node features for multiple steps.
        """
        outputs = []
        H = None  # Initialize hidden state

        # Loop through timesteps
        for i in range(steps):
            H = self.forward(x[i], edge_index, edge_weight=edge_weight, H=H, batch=batch)
            outputs.append(H)

        return torch.stack(outputs, dim=0)  # Shape: (steps, batch_size, num_nodes, features)

    def forward(self, x, edge_index, edge_weight=None, H=None, batch=None):
        """
        Forward pass with support for batching.

        Args:
            x (torch.FloatTensor): Node features (batch_size, num_nodes, features).
            edge_index (torch.LongTensor): Edge index of the graph(s).
            edge_weight (torch.FloatTensor, optional): Edge weights of the graph(s).
            H (torch.FloatTensor, optional): Hidden states for batched graphs.
            batch (torch.LongTensor, optional): Batch indices for nodes in the graph batch.

        Returns:
            torch.FloatTensor: Updated node features or hidden states.
        """
        if batch is not None:
            # Convert to dense batch representation
            x, mask = to_dense_batch(x, batch)
            x = x.view(-1, x.size(-1))
            print(x.shape)
            H = H if H is not None else torch.zeros_like(x)
        else:
            # Handle a single graph
            H = H if H is not None else torch.zeros_like(x)

        return super().forward(x, edge_index, edge_weight=edge_weight, H=H)


# Example usage for batched graphs
def test_pyg_diffusion_batched():
    # Example graph batch
    edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    # Node features for two graphs with 325 nodes each
    x1 = torch.rand((12, 325, 2))  # (seq_len, num_nodes, features)
    x2 = torch.rand((12, 325, 2))  # (seq_len, num_nodes, features)
    my_x = torch.cat([x1, x2], dim=1)  # (12, 650, 2)

    # Batch tensor for 2 graphs
    batch = torch.arange(2).repeat_interleave(325)

    # Instantiate the diffusion convolution layer
    diffusion_layer = PyGDiffusionConvolution(node_features=2, hidden_size=2, k=2)

    # Perform multi-step prediction
    multi_step_output = diffusion_layer.multi_step_prediction(
        my_x, edge_index1, steps=12, batch=batch
    )
    print("Multi-Step Output for Batched Graphs:", multi_step_output.shape)


test_pyg_diffusion_batched()

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

from torch_geometric_temporal.dataset import WindmillOutputLargeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = WindmillOutputLargeDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
model = RecurrentGCN(8).to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(200):
    model.train()
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        # print(snapshot.x.to("cuda").shape, snapshot.edge_index.to("cuda").shape, snapshot.edge_attr.to("cuda").shape)
        # exit()
        y_hat = model(snapshot.x.to("cuda"), snapshot.edge_index.to("cuda"), snapshot.edge_attr.to("cuda"))
        cost = cost + torch.mean((y_hat-snapshot.y.to("cuda")))
        # cost +=1
    # print(cost)
    cost = cost / (time+1)
    # print(cost.item())
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x.to("cuda"), snapshot.edge_index.to("cuda"), snapshot.edge_attr.to("cuda"))
        cost = cost + torch.mean(abs(y_hat-snapshot.y.to("cuda")))
    cost = cost / (time+1)
    cost = cost.item()
    print(f"Epoch {epoch} MAE: {cost}")

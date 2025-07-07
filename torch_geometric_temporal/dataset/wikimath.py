import json
import urllib
import numpy as np
from ..signal import StaticGraphTemporalSignal


class WikiMathsDatasetLoader(object):
    """A dataset of vital mathematics articles from Wikipedia. We made it
    public during the development of PyTorch Geometric Temporal. The
    underlying graph is static - vertices are Wikipedia pages and edges are
    links between them. The graph is directed and weighted. Weights represent
    the number of links found at the source Wikipedia page linking to the target
    Wikipedia page. The target is the daily user visits to the Wikipedia pages
    between March 16th 2019 and March 15th 2021 which results in 731 periods.
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/wikivital_mathematics.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):

        targets = []
        for time in range(self._dataset["time_periods"]):
            targets.append(np.array(self._dataset[str(time)]["y"]))
        stacked_target = np.stack(targets)
        standardized_target = (
            stacked_target - np.mean(stacked_target, axis=0)
        ) / np.std(stacked_target, axis=0)
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> StaticGraphTemporalSignal:
        """Returning the Wikipedia Vital Mathematics data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Wiki Maths dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

class seq_seq_WikiMathsDatasetLoader(object):
    """A dataset of vital mathematics articles from Wikipedia. We made it
    public during the development of PyTorch Geometric Temporal. The
    underlying graph is static - vertices are Wikipedia pages and edges are
    links between them. The graph is directed and weighted. Weights represent
    the number of links found at the source Wikipedia page linking to the target
    Wikipedia page. The target is the daily user visits to the Wikipedia pages
    between March 16th 2019 and March 15th 2021 which results in 731 periods.
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/wikivital_mathematics.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        base = np.array(self._dataset["weights"])
        print(base.shape)
        base = (base - base.min()) / (base.max() - base.min() ).T
        self._edge_weights = base
        self._edge_weights = np.array(self._dataset["weights"]).T
        # print("og: ", self._edge_weights.shape)
        # self._edge_weights = (self._edge_weights - self._edge_weights.min()) / (self._edge_weights.max() - self._edge_weights.min())
        # print(self._edge_weights.shape)

    def _get_targets_and_features(self):
        targets = []
        for time in range(self._dataset["time_periods"]):
            targets.append(np.array(self._dataset[str(time)]["y"]))
        stacked_target = np.stack(targets)

        self.mean = np.mean(stacked_target, axis=0)
        self.std = np.std(stacked_target, axis=0)
        standardized_target = (
            stacked_target - np.mean(stacked_target, axis=0)
        ) / np.std(stacked_target, axis=0)
        
        # standardized_target = stacked_target
        
        self.features = [
            standardized_target[i : i + self.lags, :]
            for i in range(len(targets) - self.lags - self.lags)
        ]
        self.features = np.expand_dims(self.features, axis=-1)
        self.targets = [
            standardized_target[i + self.lags: i + self.lags + self.lags, :]
            for i in range(len(targets) - self.lags - self.lags)
        ]
        self.targets = np.expand_dims(self.targets, axis=-1)

        # targets = []
        # for time in range(self._dataset["time_periods"]):
        #     targets.append(np.array(self._dataset[str(time)]["y"]))
        # stacked_target = np.stack(targets)

        # self.mean = np.mean(stacked_target, axis=0)
        # self.std = np.std(stacked_target, axis=0)
        # standardized_target = (stacked_target - self.mean) / (
        #     self.std
        # )
        # self.features = [
        #     standardized_target[i : i + self.lags, :]
        #     for i in range(standardized_target.shape[0] - self.lags - self.lags + 1)
        # ]
        # self.features = np.expand_dims(self.features, axis=-1)
        
        # self.targets = [
        #     standardized_target[i + self.lags: i + self.lags + self.lags, :]
        #     for i in range(standardized_target.shape[0] - self.lags - self.lags + 1)
        # ]
        # self.targets = np.expand_dims(self.targets, axis=-1)

        

    def get_dataset(self, lags: int = 8) -> StaticGraphTemporalSignal:
        """Returning the Wikipedia Vital Mathematics data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Wiki Maths dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

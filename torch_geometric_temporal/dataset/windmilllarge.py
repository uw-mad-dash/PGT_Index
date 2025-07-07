import json
import urllib
import numpy as np
from ..signal import StaticGraphTemporalSignal


class seq_seq_WindmillOutputLargeDatasetLoader(object):
    """Hourly energy output of windmills from a European country
    for more than 2 years. Vertices represent 319 windmills and
    weighted edges describe the strength of relationships. The target
    variable allows for regression tasks.
    """

    def __init__(self):
        self._read_web_data()
        

    def _read_web_data(self):
        # url = "https://graphmining.ai/temporal_datasets/windmill_output.json"
        # self._dataset = json.loads(urllib.request.urlopen(url).read().decode())
        # url = "https://graphmining.ai/temporal_datasets/windmill_output.json"
        with open("data/windmill_output.json", "r") as file:
            self._dataset = json.load(file)
         

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        
        stacked_target = np.stack(self._dataset["block"])
        
        self.mean = np.mean(stacked_target, axis=0)
        self.std = np.std(stacked_target, axis=0) + 10 ** -10
        standardized_target = (stacked_target - self.mean) / (
            self.std
        )
        self.features = [
            standardized_target[i : i + self.lags, :]
            for i in range(standardized_target.shape[0] - self.lags - self.lags + 1)
        ]
        self.features = np.expand_dims(self.features, axis=-1)
        
        self.targets = [
            standardized_target[i + self.lags: i + self.lags + self.lags, :]
            for i in range(standardized_target.shape[0] - self.lags - self.lags + 1)
        ]
        self.targets = np.expand_dims(self.targets, axis=-1)
        

    def get_dataset(self, lags: int = 8) -> StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )

        return dataset


class WindmillOutputLargeDatasetLoader(object):
    """Hourly energy output of windmills from a European country
    for more than 2 years. Vertices represent 319 windmills and
    weighted edges describe the strength of relationships. The target
    variable allows for regression tasks.
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://graphmining.ai/temporal_datasets/windmill_output.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read().decode())
   
         

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        
        stacked_target = np.stack(self._dataset["block"])
        print(stacked_target.shape)
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        print(np.array(self.features).shape)
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        print(np.array(self.targets).shape)

    def get_dataset(self, lags: int = 8) -> StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

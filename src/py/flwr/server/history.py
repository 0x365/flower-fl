# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training history."""


import pprint
from functools import reduce
from typing import Dict, List, Tuple
import numpy as np
import json

from flwr.common.typing import Scalar


class History:
    """History class for training and/or evaluation metrics collection."""

    def __init__(self) -> None:
        self.losses_distributed: List[Tuple[int, float]] = []
        self.losses_centralized: List[Tuple[int, float]] = []
        self.metrics_distributed_fit: Dict[str, List[Tuple[int, Scalar]]] = {}
        self.metrics_distributed: Dict[str, List[Tuple[int, Scalar]]] = {}
        self.metrics_centralized: Dict[str, List[Tuple[int, Scalar]]] = {}

    def add_loss_distributed(self, server_round: int, loss: float) -> None:
        """Add one loss entry (from distributed evaluation)."""
        self.losses_distributed.append((server_round, loss))

    def add_loss_centralized(self, server_round: int, loss: float) -> None:
        """Add one loss entry (from centralized evaluation)."""
        self.losses_centralized.append((server_round, loss))

    def add_metrics_distributed_fit(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries (from distributed fit)."""
        for key in metrics:
            # if not (isinstance(metrics[key], float) or isinstance(metrics[key], int)):
            #     continue  # ignore non-numeric key/value pairs
            if key not in self.metrics_distributed_fit:
                self.metrics_distributed_fit[key] = []
            self.metrics_distributed_fit[key].append((server_round, metrics[key]))

    def add_metrics_distributed(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries (from distributed evaluation)."""
        for key in metrics:
            # if not (isinstance(metrics[key], float) or isinstance(metrics[key], int)):
            #     continue  # ignore non-numeric key/value pairs
            if key not in self.metrics_distributed:
                self.metrics_distributed[key] = []
            self.metrics_distributed[key].append((server_round, metrics[key]))

    def add_metrics_centralized(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries (from centralized evaluation)."""
        for key in metrics:
            # if not (isinstance(metrics[key], float) or isinstance(metrics[key], int)):
            #     continue  # ignore non-numeric key/value pairs
            if key not in self.metrics_centralized:
                self.metrics_centralized[key] = []
            self.metrics_centralized[key].append((server_round, metrics[key]))

    def reformat_json(self, bad_array) -> {}:
        """Removes start value if one exists"""
        temp_all = np.array(bad_array)
        temp = {}
        if temp_all[0,0] == 0:
            temp.update({"rounds": temp_all[1:,1].tolist()})
            temp.update({"start": temp_all[0,1].tolist()})
        else:
            temp.update({"rounds": temp_all[:,1].tolist()})
        return temp

    def repr_json(self) -> {}:
        """Ouputs same data as __repr__ below but in json format"""
        rep = {}
        if self.losses_distributed:
            rep.update({"History (loss, distributed)": self.reformat_json(self.losses_distributed)})
        if self.losses_centralized:
            rep.update({"History (loss, centralized)": self.reformat_json(self.losses_centralized)})
        if self.metrics_distributed_fit:
            temp = {}
            for x in self.metrics_distributed_fit.keys():
                temp.update({x: self.reformat_json(self.metrics_distributed_fit[x])})
            rep.update({"History (metrics, distributed, fit)": temp})
        if self.metrics_distributed:
            temp = {}
            for x in self.metrics_distributed.keys():
                temp.update({x: self.reformat_json(self.metrics_distributed[x])})
            rep.update({"History (metrics, distributed, evaluate)": temp})
        if self.metrics_centralized:
            temp = {}
            for x in self.metrics_centralized.keys():
                temp.update({x: self.reformat_json(self.metrics_centralized[x])})
            rep.update({"History (metrics, centralized)": temp})
        return rep

    def __repr__(self) -> str:
        """Create a representation of History.

        The representation consists of the following data (for each round) if present:

        * distributed loss.
        * centralized loss.
        * distributed training metrics.
        * distributed evaluation metrics.
        * centralized metrics.

        Returns
        -------
        representation : str
            The string representation of the history object.
        """
        rep = ""
        if self.losses_distributed:
            rep += "History (loss, distributed):\n" + reduce(
                lambda a, b: a + b,
                [
                    f"\tround {server_round}: {loss}\n"
                    for server_round, loss in self.losses_distributed
                ],
            )
        if self.losses_centralized:
            rep += "History (loss, centralized):\n" + reduce(
                lambda a, b: a + b,
                [
                    f"\tround {server_round}: {loss}\n"
                    for server_round, loss in self.losses_centralized
                ],
            )
        if self.metrics_distributed_fit:
            rep += (
                "History (metrics, distributed, fit):\n"
                + pprint.pformat(self.metrics_distributed_fit)
                + "\n"
            )
        if self.metrics_distributed:
            rep += (
                "History (metrics, distributed, evaluate):\n"
                + pprint.pformat(self.metrics_distributed)
                + "\n"
            )
        if self.metrics_centralized:
            rep += "History (metrics, centralized):\n" + pprint.pformat(
                self.metrics_centralized
            )
        return rep

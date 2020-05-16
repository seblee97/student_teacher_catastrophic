from typing import Union, List, Dict, Iterator

import torch
from torchvision.datasets.mnist import MNIST


class Constants:

    TEST_DATA_TYPES = Union[
        List[Dict[str, torch.Tensor]],
        Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
        ]

    EVEN_ODD_MAPPING = {
        0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1
        }

    GREATER_FIVE_MAPPING = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1
        }

    TRAINABLE_PARAMETERS_TYPE = Union[
        List[Dict[str, Iterator[torch.nn.Parameter]]],
        Iterator[torch.nn.Parameter]
        ]

    MNIST_DATASET_TYPE = MNIST

    DATASET_TYPES = Union[List[MNIST], MNIST]

    # Hard-coded subplot layouts for different numbers of graphs
    GRAPH_LAYOUTS = {
        1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3),
        6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 3), 10: (2, 5),
        11: (3, 4), 12: (3, 4)
    }

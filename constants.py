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

    TEACHER_SHADES = ["#2A9D8F", "#E9C46A"]

    STUDENT_SHADES = ["#264653", "#E9C46A"]

    ORANGE_SHADES = [
        "#E9C46A", "#F4A261", "#E76F51",
        "#D5B942", "#D9D375", "#EDFBC1",
        "#FC9E4F", "#F17105"
        ]

    TORQUOISE_SHADES = [
        "#2A9D8F", "#4E8098", "#17301C",
        "#4B644A", "#89A894", "#1C3738",
        "#32746D", "#01200F"
        ]

    BLUE_SHADES = [
        "#5465ff", "#788bff", "#9bb1ff",
        "#bfd7ff", "#e2fdff"
        ]

    GREEN_SHADES = [
        "#143601", "#245501", "#538d22",
        "#73a942", "#aad576"
        ]

    MNIST_TRAIN_SET_SIZE = 60000
    MNIST_TEST_SET_SIZE = 10000
    MNIST_FLATTENED_DIM = 784

from typing import Union, List, Dict, Iterator

import torch


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

from abc import ABC, abstractmethod

from typing import List, Union, Tuple

class _Template(ABC):

    LEVELS: Union[List[str], str]
    OPTIONAL: List[str]

    @classmethod
    @abstractmethod
    def get_fields(cls) -> List:
        raise NotImplementedError("Base class method")

    @classmethod
    def get_levels(cls) -> Union[List[str], str]:
        return cls.LEVELS

    @classmethod
    def get_template_name(cls) -> str:
        return cls.LEVELS[-1]

    @classmethod
    def get_optional_fields(cls) -> List[str]:
        return cls.OPTIONAL

class Field:

    def __init__(self, name: str, types: List, reqs: Union[List, None]):

        self.name = name
        self.types: Tuple = tuple(types)
        self.reqs = reqs 

    def get_name(self):
        return self.name 

    def get_types(self):
        return self.types 

    def get_reqs(self):
        return self.reqs
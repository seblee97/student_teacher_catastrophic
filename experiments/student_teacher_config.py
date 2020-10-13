from typing import Union, Dict

from config_manager import base_configuration


class StudentTeacherConfiguration(base_configuration.BaseConfiguration):

    def __init__(self, config: Union[str, Dict]):
        super().__init__(config=config)
        self._validate_configuration()

    def _validate_configuration(self):
        """Method to check for non-trivial associations 
        in the configuration.
        """
        pass
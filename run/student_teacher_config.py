from typing import Union, Dict

from config_manager import base_configuration
from config_manager import config_template


class StudentTeacherConfiguration(base_configuration.BaseConfiguration):
    def __init__(self, config: Union[str, Dict], template: config_template.Template):
        super().__init__(configuration=config, template=template)
        self._validate_configuration()

    def _validate_configuration(self):
        """Method to check for non-trivial associations
        in the configuration.
        """
        pass

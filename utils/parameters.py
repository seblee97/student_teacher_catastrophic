from .base_template import _Template, Field

from typing import Any, List, Union, Dict, Type, overload

import yaml
import os
import collections
import six
import warnings
import inspect

ParameterGetTypes = Union[List[str], List[int], int, str, bool, float]

class Parameters(object):
    
    def __init__(self, parameters: Dict[str, Any]):
        self._config = parameters
    
    @overload
    def get(self, property_name: str): ...

    @overload
    def get(self, property_name: List[str]): ...

    def get(self, property_name):
        """
        Return value associated with property_name in configuration

        :param property_name: name of parameter in configuration. 
                              Could be list if accessing nested part of dict.
        :return: value associated with property_name
        """
        if isinstance(property_name, list):
            value = self._config
            for prop in property_name:
                value = value.get(prop, "Unknown Key")
                if value == "Unknown Key":
                    raise ValueError("Config key {} unrecognised".format(prop))
            return value
        elif isinstance(property_name, str):
            value = self._config.get(property_name, "Unknown Key")
            if value == "Unknown Key":
                raise ValueError("Config key {} unrecognised".format(property_name))
            else:
                return value
        else:
            raise TypeError("property_name supplied has wrong type. Must be list of strings or string.")

    def get_property_description(self, property_name: str) -> str:
        """
        Return description of configuration property

        :param property_name: name of parameter to query for description
        :return: description of property in configuration
        """
        raise NotImplementedError # TODO: Is this worth doing? .yaml not particularly amenable 

    def _validate_field(self, field: Field, data: Dict, level: str) -> None:
        field_name = field.get_name()
        allowed_field_types = field.get_types()
        additional_reqs = field.get_reqs()

        # ensure field exists
        assert field_name in data, "{} not specified in configuration at level {}".format(field_name, level)

        # ensure value give for field is correct type
        field_value = data[field_name]
        assert isinstance(field_value, allowed_field_types), \
        "Type given for field {} at level {} in config is {}. Must be one of {}".format(field_name, level, type(field_value), allowed_field_types)

        # ensure other requirements are satisfied
        if additional_reqs:
            for r, requirement in enumerate(additional_reqs):
                assert requirement(field_value), "Additional requirement check {} for field {} failed".format(r, field_name)

        print("Validating field: {} at level {} in config...".format(field_name, level))

    def check_template(self, template: _Template):
        template_attributes = template.get_fields()

        template_nesting = template.get_levels()
        data: Union[Dict[str, Any], Any] = self._config
        if template_nesting is "ROOT":
            level_name = "ROOT"
        else:
            level_name = '/'.join(template_nesting)
            for level in template_nesting:
                data = data.get(level)

        fields_to_check = list(data.keys())
        optional_fields = template.get_optional_fields()

        for template_attribute in template_attributes:
            if (inspect.isclass(template_attribute) and issubclass(template_attribute, _Template)):
                self.check_template(template_attribute)
                fields_to_check.remove(template_attribute.get_template_name())
            else:
                self._validate_field(field=template_attribute, data=data, level=level_name)
                fields_to_check.remove(template_attribute.get_name())

        for optional_field in optional_fields:
            if optional_field in fields_to_check:
                fields_to_check.remove(optional_field)
                warnings.warn("Optional field {} provided but NOT checked".format(optional_field))

        assert not fields_to_check, \
            "There are fields at level {} of config that have not been validated: {}".format(level_name, ", ".join(fields_to_check))

    def set_property(self, property_name: Union[str, List[str]], property_value: Any, property_description: str=None) -> None:
        """
        Add to the configuration specification

        :param property_name: name of parameter to append to configuration
        :param property_value: value to set for property in configuration
        :param property_description (optional): description of property to add to configuration
        """
        if property_name in self._config:
            raise Exception("This field is already defined in the configuration. Use ammend_property method to override current entry")
        else:
            if isinstance(property_name, str):
                self._config[property_name] = property_value
            else:
                raise TypeError("Property name type {} not assignable".format(type(property_name)))

    def ammend_property(self, property_name: str, property_value: Any, property_description: str=None) -> None:
        """
        Add to the configuration specification

        :param property_name: name of parameter to ammend in configuration
        :param property_value: value to ammend for property in configuration
        :param property_description (optional): description of property to add to configuration
        """
        raise NotImplementedError
        if property_name not in self._config:
            raise Exception("This field is not defined in the configuration. Use set_property method to add this entry")
        else:
            self._config[property_name] = property_value

    def show_all_parameters(self) -> None:
        """
        Prints entire configuration
        """ 
        print(self._config)

    def save_configuration(self, save_path: str) -> None:
        """
        Saves copy of configuration to specified path. Particularly useful for keeping track of different experiment runs

        :param save_path: path to folder in which to save configuration
        """
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(self._config, f)

    def update(self, specific_params: Dict) -> None:
        """
        Update parameter entries based on entried in specific_params.

        specific_params could be nested dictionary
        """
        def update_dict(original_dictionary, update_dictionary):
            for key, value in six.iteritems(update_dictionary):
                sub_dict = original_dictionary.get(key, {})
                if not isinstance(sub_dict, collections.Mapping): # no more nesting
                    original_dictionary[key] = value
                elif isinstance(value, collections.Mapping):
                    original_dictionary[key] = update_dict(sub_dict, value) # more nesting, recurse
                else:
                    original_dictionary[key] = value

            return original_dictionary
        
        self._config = update_dict(self._config, specific_params)
            

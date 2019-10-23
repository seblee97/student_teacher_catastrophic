from typing import Any, List, Union

import yaml
import os
import collections
import six

class StudentTeacherParameters(object):
    
    def __init__(self, parameters):
        self._config = parameters
    
    def get(self, property_name: Union[str, List[str]]) -> Any:
        """
        Return value associated with property_name in configuration

        :param property_name: name of parameter in configuration. 
                              Could be list if accessing nested part of dict.
        :return: value associated with property_name
        """
        if type(property_name) == list:
            value = self._config
            for prop in property_name:
                value = value.get(prop, "Unknown Key")
                if value == "Unknown Key":
                    raise ValueError("Config key {} unrecognised".format(prop))
            return value
        elif type(property_name) == str:
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
            if type(property_name) == str:
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

    def update(self, specific_params: dict) -> None:
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
            

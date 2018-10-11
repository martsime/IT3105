import sys
import json
import argparse
from copy import deepcopy


class Settings:
    def __init__(self, dictionary):
        self.update(dictionary)

    def update(self, dictionary):
        for key in dictionary.keys():
            value = dictionary[key]
            if type(value) == dict:
                if hasattr(self, key) and type(self.__getattribute__(key)) == Settings:
                    self.__getattribute__(key).update(value)
                else:
                    self.__setattr__(key, Settings(value))
            elif type(value) == list:
                new_value = []
                for object in value:
                    if type(object) == dict:
                        if key == 'hidden_layers' and hasattr(self, 'default_layer'):
                            base_layer = deepcopy(self.default_layer)
                            base_layer.update(object)
                            new_value.append(base_layer)
                        else:
                            new_value.append(Settings(object))
                    elif type(object) == Settings:
                        object.update(value)
                        new_value.append(object)
                    else:
                        new_value.append(object)
                self.__setattr__(key, new_value)
            else:
                self.__setattr__(key, value)


def load_settings():
    parser = argparse.ArgumentParser(description='Build an artificial neural net from a specified settings file')
    parser.add_argument('settings_file', help='Path to settings file for the network')
    args = parser.parse_args()
    filename = args.settings_file

    with open('settings/default.json') as json_file:
        try:
            settings = Settings(json.load(json_file))
        except json.decoder.JSONDecodeError as error:
            print("ERROR loading settings/default.json")
            print("Message: " + str(error))
            sys.exit(1)

    with open(filename) as json_file:
        try:
            settings.update(json.load(json_file))
        except json.decoder.JSONDecodeError as error:
            print("ERROR loading " + filename)
            print("Message: " + str(error))
            sys.exit(1)

    return settings


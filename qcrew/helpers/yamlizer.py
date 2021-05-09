"""
This module supports serializing and deserializing of those Python classes in qcrew
that inherit from Yamlable and implement the `yaml_map` property.
TODO WRITE DOCUMENTATION
"""
from abc import ABCMeta, abstractmethod
import yaml

YAML_TAG_PREFIX = u"!"


# use scientific notation if abs(value) >= threshold
def sci_not_representer(dumper, value):
    """ """
    threshold = 1e3  # arbitrarily set
    yaml_float_tag = u"tag:yaml.org,2002:float"
    value_in_sci_not = "{:.7E}".format(value) if abs(value) >= threshold else str(value)
    return dumper.represent_scalar(yaml_float_tag, value_in_sci_not)


# lists must be always represented in flow style, not block style
def sequence_representer(dumper, value):
    """ """
    yaml_seq_tag = u"tag:yaml.org,2002:seq"
    return dumper.represent_sequence(yaml_seq_tag, value, flow_style=True)


class YamlableMetaclass(ABCMeta):
    """ """

    def __init__(cls, name, bases, kwds):
        super(YamlableMetaclass, cls).__init__(name, bases, kwds)

        # set a consistent format for subclass yaml tags
        cls.yaml_tag = YAML_TAG_PREFIX + name

        # register safe loader and safe dumper
        cls.yaml_loader, cls.yaml_dumper = yaml.SafeLoader, yaml.SafeDumper

        # custom constructor and representer for Yamlable objects
        cls.yaml_loader.add_constructor(cls.yaml_tag, cls.from_yaml)
        cls.yaml_dumper.add_representer(cls, cls.to_yaml)
        # customise dumper to represent float values in scientific notation
        cls.yaml_dumper.add_representer(float, sci_not_representer)
        # customise dumper to represent tuples and lists in flow style
        cls.yaml_dumper.add_representer(list, sequence_representer)


class Yamlable(metaclass=YamlableMetaclass):
    """ """

    @property
    @abstractmethod
    def yaml_map(self):
        """ """

    @classmethod
    def from_yaml(cls, loader, node):
        """ """
        yaml_map = loader.construct_mapping(node)
        return cls(**yaml_map)

    @classmethod
    def to_yaml(cls, dumper, data):
        """ """
        return dumper.represent_mapping(data.yaml_tag, data.yaml_map)
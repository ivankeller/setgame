import json
from enum import Enum
from dataclasses import dataclass, asdict


class Attribute(Enum):
    NUMBER = 'number'
    COLOR = 'color'
    SHAPE = 'shape'
    SHADING = 'shading'


class Number(Enum):
    ONE = '1'
    TWO = '2'
    THREE = '3'


class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    PURPLE = 'purple'


class Shape(Enum):
    DIAMOND = 'diamond'
    SQUIGGLE = 'squiggle'
    OVAL = 'oval'


class Shading(Enum):
    SOLID = 'solid'
    STRIPED = 'striped'
    OPEN = 'open'


@dataclass
class Label:
    number: Number = None
    color: Color = None
    shape: Shape = None
    shading: Shading = None

    def as_dict(self):
        label_dict = asdict(self)
        for attribute, value in label_dict.items():
            label_dict[attribute] = value.value
        return label_dict

    def as_json(self):
        label_dict = self.as_dict()
        return json.dumps(label_dict)



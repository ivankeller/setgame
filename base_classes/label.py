from enum import Enum
from dataclasses import dataclass, asdict


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
    number: Number
    color: Color
    shape: Shape
    shading: Shading

    def as_dict(self):
        label_dict = asdict(self)
        for attribute, value in label_dict.items():
            label_dict[attribute] = value.value
        return label_dict


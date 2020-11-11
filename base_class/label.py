from enum import Enum
from dataclasses import dataclass


class Color(Enum):
    RED = 'r'
    GREEN = 'g'
    PURPLE = 'p'


class Number(Enum):
    ONE = '1'
    TWO = '2'
    THREE = '3'


class Shape(Enum):
    DIAMOND = 'd'
    SQUIGGLE = 's'
    OVAL = 'o'


class Shading(Enum):
    SOLID = 's'
    STRIPED = 't'
    OPEN = 'o'


@dataclass
class Label:
    color: Color
    number: Number
    shape: Shape
    shading: Shading

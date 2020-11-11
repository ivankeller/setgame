from IPython.display import display
from ipywidgets import ToggleButtons, Dropdown, HTML, HBox, Output
from typing import List, Tuple
from base_class.label import Color, Number, Shape, Shading, Label


def annotate(examples: List[str]):
    """
    Build an interactive widget for annotating Set cards images.

    Parameters
    ----------
    examples : List[str]
        paths to images

    Return
    ------
    annotations : List[Tuple[str, dict]]
        list of annotated examples (path, labels)

    """
    annotations = []
    current_index = -1



def buttons():
    color_buttons = ToggleButtons(
        options=[('red', Color.RED), ('green', Color.GREEN), ('purple', Color.PURPLE)],
        value=None,
        description='Colour:',
        button_style=''
    )
    number_buttons = ToggleButtons(
        options=[('1', Number.ONE), ('2', Number.TWO), ('3', Number.THREE)],
        value=None,
        description='Number:',
        button_style=''
    )
    shape_buttons = ToggleButtons(
        options=[('diamond', Shape.DIAMOND), ('squiggle', Shape.SQUIGGLE), ('oval', Shape.OVAL)],
        value=None,
        description='Shape:',
        button_style=''
    )
    shading_buttons = ToggleButtons(
        options=[('solid', Shading.SOLID), ('striped', Shading.STRIPED), ('open', Shading.OPEN)],
        value=None,
        description='Shading:',
        button_style=''
    )
    display(color_buttons)
    display(number_buttons)
    display(shape_buttons)
    display(shading_buttons)
    return color_buttons, number_buttons, shape_buttons, shading_buttons

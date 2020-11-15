from enum import Enum
from IPython.display import display, Image
from ipywidgets import Button, ToggleButtons, HTML, Output
from typing import List
from base_class.label import Color, Number, Shape, Shading, Label


class ButtonName(Enum):
    NUMBER = 'Number'
    COLOR = 'Color'
    SHAPE = 'Shape'
    SHADING = 'Shading'
    SUBMIT = 'Submit'


class Annotator:
    """
    Build an interactive widget for annotating Set cards images.

    Attributes
    ----------
    examples : List[str] or str
        paths to images or to directory containing images
    """

    def __init__(self, examples):
        self.examples = Annotator.list_examples(examples)
        self.cursor = -1
        self.annotations = []
        self.progress_message = HTML()
        self.output_message = Output()
        self.output_image = Output()
        self.buttons = self.make_buttons()
        self.label_buttons = self.list_label_buttons()
        self.submit_button = self.buttons[ButtonName.SUBMIT.value]

    def annotate(self):
        self.display_label_buttons()
        display(self.submit_button)
        display(self.output_message)

    def set_progress_message(self):
        nb_annotations = len(self.annotations)
        nb_remaining = len(self.examples) - self.cursor
        self.progress_message.value = f'{nb_annotations} examples annotated, {nb_remaining} remaining'

    @staticmethod
    def list_examples(examples):
        pass

    def show_next_example(self):
        pass

    def list_label_buttons(self):
        label_buttons = [
            ButtonName.NUMBER.value,
            ButtonName.COLOR.value,
            ButtonName.SHAPE.value,
            ButtonName.SHADING.value
        ]
        return [button for button_name, button in self.buttons.items() if button_name in label_buttons]

    def make_buttons(self):
        number_button = ToggleButtons(
            options=[('1', Number.ONE.value), ('2', Number.TWO.value), ('3', Number.THREE.value)],
            value=None,
            description=ButtonName.NUMBER.value,
            #button_style=''
        )
        color_button = ToggleButtons(
            options=[('red', Color.RED.value), ('green', Color.GREEN.value), ('purple', Color.PURPLE.value)],
            value=None,
            description=ButtonName.COLOR.value,
            button_style=''
        )
        shape_button = ToggleButtons(
            options=[('oval', Shape.OVAL), ('diamond', Shape.DIAMOND), ('squiggle', Shape.SQUIGGLE)],
            value=None,
            description=ButtonName.SHAPE.value,
            button_style=''
        )
        shading_button = ToggleButtons(
            options=[('open', Shading.OPEN), ('striped', Shading.STRIPED), ('solid', Shading.SOLID)],
            value=None,
            description=ButtonName.SHADING.value,
            button_style=''
        )
        submit_button = Button(description=ButtonName.SUBMIT.value)
        buttons = {
            number_button.description: number_button,
            color_button.description: color_button,
            shape_button.description: shape_button,
            shading_button.description: shading_button,
            submit_button.description: submit_button,
        }
        return buttons

    def display_label_buttons(self):
        for button in self.label_buttons:
            display(button)

    def remove_label_buttons_values(self):
        for button in self.label_buttons:
            button.value = None

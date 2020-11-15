from enum import Enum
from IPython.display import display, Image
from ipywidgets import Button, ToggleButtons, HTML, Output
from typing import List
from base_class.label import Color, Number, Shape, Shading, Label
from utils import list_images_in_directory


class ButtonName(Enum):
    NUMBER = 'Number'
    COLOR = 'Color'
    SHAPE = 'Shape'
    SHADING = 'Shading'
    SUBMIT = 'Submit'


class Annotator:
    """
    Interactive Ipython widget for annotating Set cards images.

    Attributes
    ----------
    examples : List[str] or str
        if list of strings: list of paths to image to annotate
        if str: path to a directory containing images to annotate (jpg or png)

    """

    def __init__(self, examples):
        self.examples = Annotator.list_examples_to_annotate(examples)
        self.cursor = -1
        self.annotations = []
        self.progress_message = HTML()
        self.output_message = Output()
        self.output_image = Output()
        self.buttons = self.make_all_buttons()
        self.label_buttons = self.list_label_buttons()
        self.submit_button = self.buttons[ButtonName.SUBMIT.value]

    def annotate(self):
        """Run the annotation widgets"""
        self.set_progression_message()
        display(self.progress_message)
        display(self.output_image)
        self.initialize_label_buttons()
        self.display_all_buttons()
        display(self.output_message)
        self.show_next_example()
        self.submit_button.on_click(lambda button: self.on_button_clicked(button))

    def set_progression_message(self):
        nb_annotations = len(self.annotations)
        nb_remaining = len(self.examples) - self.cursor
        self.progress_message.value = f'{nb_annotations} examples annotated, {nb_remaining} example remaining'

    @staticmethod
    def list_examples_to_annotate(examples):
        """
        Return list path to images to annotate.

        Parameters
        ----------
        examples : List[str] or str
            if list of strings: list of paths to image to annotate
            if str: path to a directory containing images to annotate (wiht extension: jpg or png)

        Returns
        -------
        List[str] of path to images to annotate

        """
        # expected directory
        if isinstance(examples, str):
            return list_images_in_directory(examples)
        # expected list of paths
        elif isinstance(examples, List):
            return examples
        else:
            raise TypeError("Wrong argument type.")

    def list_label_buttons(self):
        label_buttons = [
            self.buttons[ButtonName.NUMBER.value],
            self.buttons[ButtonName.COLOR.value],
            self.buttons[ButtonName.SHAPE.value],
            self.buttons[ButtonName.SHADING.value]
        ]
        return label_buttons

    @staticmethod
    def make_all_buttons():
        """Build all necessary buttons.

        Return
        ------
        buttons : dict
            key: str, buttons description name ('Color', 'Number', 'Submit', etc.)
            value: Buttons or ToggleButtons

        """
        number_button = ToggleButtons(
            options=[('1', Number.ONE.value), ('2', Number.TWO.value), ('3', Number.THREE.value)],
            description=ButtonName.NUMBER.value,
        )
        color_button = ToggleButtons(
            options=[('red', Color.RED.value), ('green', Color.GREEN.value), ('purple', Color.PURPLE.value)],
            description=ButtonName.COLOR.value,
        )
        shape_button = ToggleButtons(
            options=[('oval', Shape.OVAL.value), ('diamond', Shape.DIAMOND.value), ('squiggle', Shape.SQUIGGLE.value)],
            description=ButtonName.SHAPE.value,
        )
        shading_button = ToggleButtons(
            options=[('open', Shading.OPEN.value), ('striped', Shading.STRIPED.value), ('solid', Shading.SOLID.value)],
            description=ButtonName.SHADING.value,
        )
        submit_button = Button(description=ButtonName.SUBMIT.value)

        buttons = {
            ButtonName.NUMBER.value: number_button,
            ButtonName.COLOR.value: color_button,
            ButtonName.SHAPE.value: shape_button,
            ButtonName.SHADING.value: shading_button,
            ButtonName.SUBMIT.value: submit_button
        }
        return buttons

    def display_all_buttons(self):
        for button in self.buttons.values():
            display(button)

    def initialize_label_buttons(self):
        for button in self.label_buttons:
            button.value = None

    def disable_all_buttons(self):
        for button in self.buttons.values():
            button.disabled = True

    def show_next_example(self):
        self.cursor += 1
        self.set_progression_message()
        if self.cursor >= len(self.examples):
            self.disable_all_buttons()
            with self.output_message:
                print('Annotation completed.')
            return
        with self.output_image:
            self.output_image.clear_output()
            display(Image(self.examples[self.cursor], width=200))

    def on_button_clicked(self, but):
        responses = self.get_label_buttons_response()
        missing_attributes = Annotator.get_missing_attributes(responses)
        self.output_message.clear_output()
        if missing_attributes:
            with self.output_message:
                print(f"Missing value for {missing_attributes}. Retry.")
        else:
            with self.output_message:
                annotation = ''.join([response for response in list(responses.values())])
                self.annotations.append((self.examples[self.cursor], annotation))
                print(f"Annotation submitted: {annotation}")
            self.initialize_label_buttons()
            self.show_next_example()

    def get_label_buttons_response(self):
        """Return a dictionary of the button values.

        Returns
        -------
        dict
            key: buttons description
            value: button value, can be None if missing response

        """
        return {button.description: button.value for button in self.label_buttons}

    @staticmethod
    def get_missing_attributes(responses):
        """Return the list of button's name that without response."""
        missing = []
        for attribute, response in responses.items():
            if response is None:
                missing.append(attribute)
        return missing

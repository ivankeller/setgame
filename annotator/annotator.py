import os
import json
from loguru import logger
from enum import Enum
from IPython.display import display, Image
from ipywidgets import Button, ToggleButtons, HTML, Output
from typing import List, Dict, Union
from setgame.base_classes.label import Attribute, Color, Number, Shape, Shading
from setgame.utils import list_images_in_directory


class ButtonName(Enum):
    NUMBER = Attribute.NUMBER.value
    COLOR = Attribute.COLOR.value
    SHAPE = Attribute.SHAPE.value
    SHADING = Attribute.SHADING.value
    SUBMIT = 'SUBMIT'


DEFAULT_OUTPUT_SUBDIR = 'labels'


class Annotator:
    """
    Interactive Ipython widget for annotating Set cards images.
    Annotations are iteratively saved as json files.

    Attributes
    ----------
    examples : List[str] or str
        if list of strings: list of paths to image to annotate
        if str: path to a directory containing images to annotate (jpg or png)
    output_dir : str
        directory to save annotations, if None (default) save to default sub-directory

    """

    def __init__(self, examples, output_dir=None):
        self.examples = Annotator.list_examples_to_annotate(examples)
        self.output_dir = self.set_output_dir(output_dir)
        self.cursor = 0
        self.annotations = []
        self.progress_message = HTML()
        self.output_message = Output()
        self.output_image = Output()
        self.label_buttons, self.submit_button = self.make_all_buttons()

    def annotate(self):
        """Run the annotation widgets."""
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
        self.progress_message.value = f'{nb_annotations} example(s) annotated, {nb_remaining} example(s) remaining'

    def set_output_dir(self, output_dir: str) -> str:
        """Set output directory. If None set to default sub-directory."""
        if output_dir is None:
            parent_dir = os.path.dirname(self.examples[0])
            output_dir = os.path.join(parent_dir, DEFAULT_OUTPUT_SUBDIR)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'created output directory: {output_dir}')
        return output_dir

    @staticmethod
    def list_examples_to_annotate(examples: Union[List[str], str]) -> List[str]:
        """
        Return list of paths to images to annotate.

        Parameters
        ----------
        examples : List[str] or str
            if list of strings: list of paths to image to annotate
            if str: path to a directory containing images to annotate (with extension: jpg or png)

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

    @staticmethod
    def make_all_buttons() -> (Dict[str, ToggleButtons], Button):
        """Build all necessary buttons.

        Return
        ------
        buttons : (dict of {str: ToggleButtons}, Button)
            label buttons (for number, color, shape and shading) and submit button

        """
        number_button = ToggleButtons(
            options=[('1', Number.ONE), ('2', Number.TWO), ('3', Number.THREE)],
            description=ButtonName.NUMBER.value,
        )
        color_button = ToggleButtons(
            options=[('red', Color.RED), ('green', Color.GREEN), ('purple', Color.PURPLE)],
            description=ButtonName.COLOR.value,
        )
        shape_button = ToggleButtons(
            options=[('oval', Shape.OVAL), ('diamond', Shape.DIAMOND), ('squiggle', Shape.SQUIGGLE)],
            description=ButtonName.SHAPE.value,
        )
        shading_button = ToggleButtons(
            options=[('open', Shading.OPEN), ('striped', Shading.STRIPED), ('solid', Shading.SOLID)],
            description=ButtonName.SHADING.value,
        )
        submit_button = Button(description=ButtonName.SUBMIT.value)

        label_buttons = {
            number_button.description: number_button,
            color_button.description: color_button,
            shape_button.description: shape_button,
            shading_button.description: shading_button,
        }
        return label_buttons, submit_button

    def display_all_buttons(self):
        for button in self.label_buttons.values():
            display(button)
        display(self.submit_button)

    def initialize_label_buttons(self):
        for button in self.label_buttons.values():
            button.value = None

    def disable_all_buttons(self):
        for button in self.label_buttons.values():
            button.disabled = True
        self.submit_button.disabled = True

    def show_next_example(self):
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
        responses = self.get_label_buttons_responses()
        missing_attributes = Annotator.get_missing_attributes(responses)
        self.output_message.clear_output()
        if missing_attributes:
            with self.output_message:
                print(f"Missing value for {missing_attributes}. Retry.")
        else:
            with self.output_message:
                annotation = Annotator.get_annotation_as_json_string(responses)
                current_example = self.examples[self.cursor]
                self.annotations.append((current_example, annotation))
                self.save_annotation(annotation, current_example)
                print(f"Annotation submitted: {annotation}")
            self.initialize_label_buttons()
            self.cursor += 1
            self.show_next_example()

    def get_label_buttons_responses(self) -> Dict[str, Union[Number, Color, Shape, Shading]]:
        """Return label buttons values."""
        return {att: button.value for att, button in self.label_buttons.items()}

    @staticmethod
    def get_annotation_as_json_string(response: Dict[str, Union[Number, Color, Shape, Shading]]) -> str:
        """Get label buttons responses as json string."""
        annotation = {att: button_response.value for att, button_response in response.items()}
        return json.dumps(annotation)

    def save_annotation(self, annotation: str, example: str) -> None:
        """Save annotation as json file"""
        basename = os.path.splitext(os.path.basename(example))[0]
        destination_path = os.path.join(self.output_dir, basename + '.json')
        with open(destination_path, 'w') as destination:
            destination.write(annotation + '\n')
            logger.info(f'annotation saved to {destination_path}')

    @staticmethod
    def get_missing_attributes(responses: Dict[str, Union[Number, Color, Shape, Shading]]) -> List[str]:
        """Return the list of labels button's name without response."""
        missing = []
        for att, response in responses.items():
            if response is None:
                missing.append(att)
        return missing

# setgame

## 0.What is this project about?
The goal of this project is to build an application that automatically finds the solution(s) for the Set game (see https://en.wikipedia.org/wiki/Set_(card_game)).  

From an image of a board of cards it should detect correct sets of 3 cards.

For example, given the left picture as input we expect the output looks like in the right picture with the yellow rectangles indicating one solution (we could also output all the solutions):  
<img src="https://user-images.githubusercontent.com/8144090/98802392-8d633600-2413-11eb-96a3-14331dae4dd6.jpg" width=400> 
<img src="https://user-images.githubusercontent.com/8144090/98802503-b5529980-2413-11eb-887b-626108971137.jpg" width=400>

This is mainly a computer vision task consisting of identifying each card from a picture of a board of cards.  
Given the identification of the cards all the solutions can then easily be found with brute force.  

A card has 4 attributes. Each attribute can take 3 values:
- **number** (of shapes in one card): 1, 2 or 3
- **color**: red, green or purple
- **shape**: oval, diamond or squiggle
- **shading**: open, striped or solid

### 0.1 Main steps to achieve the task
1. segment all the individual cards from the image of the board of cards
2. identify each card: run a multi-labels image classifier (or one classifier per card attribute)
3. solve the Set game problem (brute force)
4. display the solution(s) on the original input image of the board of cards (ex: draw rectangles around the solution cards)

### 0.2 Approach to build the system
1. make a custom image segmentator: from a image of a board of cards with arbitrary size, it should output the images of each individual card. Done
2. train an image classifier to identify the values of the 4 attributes of a card:
    * prepare the dataset:
        * make photos of boards taken in different settings (background, lighting, etc.) with all the different cards. Done.
        * use the image segmentator built in 1. to obtain a set of numerous and diverse individual card images examples. Done.
        * build a tool for annotating card images. Done: Setcard_annotator https://github.com/ivankeller/setcard_annotator
        * manually label the cards images using Setcard_annotator
        * perform image augmentation to enrich the dataset (rotation, scaling, color, etc.)
        * normalize images: size, white balance, etc.
    * train a multi-labels card classifier or 4 classifiers (one for each card attribute).
3. Wrap all together into an application.
4. Bonus: make the application to run live from a smartphone camera (Android or iOS)

## 1. Setup

### Install uv (package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create the environment and install dependencies
From the `setgame/` directory:
```bash
uv sync
```
This creates a `.venv/` virtualenv and installs all dependencies declared in `pyproject.toml`, including PyTorch, timm, and OpenCV.

## Run the tests
```bash
uv run python -m unittest discover -v
```

## 2. Segment cards from board images
Segment cards from a single board image (JPEG or PNG):
```bash
uv run python setgame/segmentboard/scripts/segment_cards_from_board_image.py \
    -i <input board image> \
    -o <output directory>
```
Segment all board images in a directory by editing `INPUT_DIR` and `OUTPUT_DIR` in the script, then running:
```bash
uv run python setgame/segmentboard/scripts/segment_all_board_images.py
```

## 3. Train the card attribute classifier

The classifier is a multi-task EfficientNet-B0 model with four output heads (number, shape, shading, color). Training uses two phases: backbone frozen (head warm-up), then full fine-tuning with differential learning rates.

### Data layout expected
```
data/
  1_segmented_cards/   ← original segmented card JPEGs
  2_labels/            ← JSON label files (one per card image)
  3_augmented/         ← aug0/, aug1/, … subdirectories of augmented images
  models/              ← trained weights will be saved here
```

Label JSON format:
```json
{"number": "2", "color": "purple", "shape": "squiggle", "shading": "striped"}
```

### Run training
```bash
uv run python scripts/train_card_classifier.py
```

Default paths point to `../data/` relative to the `setgame/` directory. Override any argument as needed:
```bash
uv run python scripts/train_card_classifier.py \
    --originals  ../data/1_segmented_cards \
    --labels     ../data/2_labels \
    --augmented  ../data/3_augmented \
    --output     ../data/models/set_card_classifier.pth \
    --phase1-epochs 5 \
    --phase2-epochs 20 \
    --batch-size 32
```

Pass `--augmented ""` to train on original images only (no augmentation).

The best model (lowest validation loss) is automatically saved to `--output`.

### Expected training results
On the provided dataset (~220 originals + 4560 augmented images, MPS/GPU):

| Attribute | Val accuracy (≈ epoch P2-06) |
|-----------|------------------------------|
| shape     | 100%                         |
| shading   | 100%                         |
| color     | 100%                         |
| number    | ~90%                         |

## 4. Solve a board (full pipeline)

Given a photo of a board of Set cards, this script segments the cards, classifies their attributes, finds a valid Set, and outputs the original image annotated with red rectangles around the three solution cards.

```bash
uv run python scripts/solve_board.py \
    -i path/to/board.jpg \
    -m ../data/models/set_card_classifier.pth \
    -o path/to/output.jpg
```

All arguments:
```
-i / --input            Board image path (JPG or PNG)          [required]
-m / --model            Classifier weights path (.pth)         [required]
-o / --output           Save annotated image to this path      [optional]
--no-display            Skip opening the result in the viewer
--background-thres      Segmentation threshold (default: 0.25)
```

The result image is also opened automatically in the system image viewer unless `--no-display` is passed.

From a Python script or notebook:
```python
from setgame.scripts.solve_board import solve

result = solve(
    board_path  = "path/to/board.jpg",
    model_path  = "../data/models/set_card_classifier.pth",
    output_path = "path/to/output.jpg",
)
# result is a PIL.Image.Image
```

## 5. Predict card attributes

Use `predict_card` or `predict_card_from_path` from `setgame.classify_card.predict`:

```python
from PIL import Image
from setgame.classify_card.predict import load_model, predict_card

# Load trained model
model = load_model("../data/models/set_card_classifier.pth")

# Predict from a PIL image (e.g. after segmentation)
image = Image.open("my_card.jpg").convert("RGB")
attrs = predict_card(image, model)
# → {"number": "2", "shape": "oval", "shading": "striped", "color": "red"}

# Or directly from a file path
from setgame.classify_card.predict import predict_card_from_path
attrs = predict_card_from_path("my_card.jpg", model)
```

Integrate with the existing `Label` dataclass:
```python
from setgame.base_classes.label import Label, Number, Color, Shape, Shading

label = Label(
    number  = Number(attrs["number"]),
    color   = Color(attrs["color"]),
    shape   = Shape(attrs["shape"]),
    shading = Shading(attrs["shading"]),
)
```

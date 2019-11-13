# setgame

Detect correct sets of three cards in Set game from an image of a board (table) of cards (see https://en.wikipedia.org/wiki/Set_(card_game))

  * prepare train/test data sets
     * segment a board of cards image into individual card images
     * manually label cards (set up a web app for labeling)
  * train a multi-labels card classifier: four attributes per card
  * from a board of cards identify all cards and all possible "sets".

## Setup

### Environment
from the root project directory:
```bash
pipenv install
pipenv shell
```

## Test 
```bash
python -m unittest discover -v
```

## Segment cards from board images  
From the root directory of this project do:
```bash
export PYTHONPATH=.
```
Segment cards from a single board image:
```bash
./scripts/segment_cards_from_board_image.py <input board image> <output directory>
```
Segment cards for all board images in a given directory:  
- edit file `./scripts/segment_all_board_images.py` adapting paths
- run it:
    ```bash
    ./scripts/segment_all_board_images.py
    ```




# setgame

Detect correct sets of three cards in Set game from an image of a board (table) of cards (see https://en.wikipedia.org/wiki/Set_(card_game))

  * prepare train/test data sets
     * segment a board of cards image into individual card images
     * manually label cards (set up a web app for labeling)
  * train a multi-labels card classifier: four attributes per card
  * from a board of cards identify all cards and all possible "sets".

## setup

### environment
from the root project directory:
```bash
pipenv install
pipenv shell
```

## test 
```bash
python -m unittest discover -v
```

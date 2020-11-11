# setgame
The goal of this project is to build an application that automatically finds the solution(s) for the Set game (see https://en.wikipedia.org/wiki/Set_(card_game)).  

From an image of a board of cards it should detect correct sets of 3 cards.

For example, given the left picture as input we expect the output like in the right picture with the yellow rectangles indicating one solution:  
<img src="https://user-images.githubusercontent.com/8144090/98802392-8d633600-2413-11eb-96a3-14331dae4dd6.jpg" width=400> 
<img src="https://user-images.githubusercontent.com/8144090/98802503-b5529980-2413-11eb-887b-626108971137.jpg" width=400>

This is mainly a computer vision task consisting of identifying each card from a picture of a board of cards.  
Given the identification of the cards all the solutions can then easily be found by brute force.  

A card has 4 attributes. Each attribute can take 3 values:
- **number of shapes**: 1, 2 or 3
- **color**: red, green or purple
- **shape**: diamond, squiggle or oval
- **shading**: solid, striped or open

## Steps to achieve the task
1. segment each card from the image of the board of cards
2. identify each card: run a multi-labels image classifier (or one classifier per card attribute)
3. solve the Set game problem (brute force)
4. display the solution on the original input image of the board of cards (ex: draw rectangles around the solution cards)

## Approach to build the system
1. make a custom image segmenter: from a image of a board of cards with arbitrary size, outputs the images of each individual card (done).
2. train an image classifier to identify the values of the 4 attributes of a card:
    * prepare the dataset
        * use the image segmenter to get a set of numerous individual card images from several pictures of board taken in different settings (background, lighting, etc.)
        * manually label the cards (maybe set up a web app for labeling, or find another solution for image annotation)
        * perform image augmentation to enrich the dataset (rotation, scaling, color, etc.)
    * train a multi-labels card classifier or 4 classifiers, one for each card attribute.
3. Wrap all together into an application.
4. Bonus: make the application to run on a smartphone (Android)

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
Segment cards from a single board image (JPEG or PNG):
```bash
./scripts/segment_cards_from_board_image.py -i <input board image> -o <output directory>
```
Segment cards for all board images in a given directory:  
- edit file `./scripts/segment_all_board_images.py` adapting paths
- run it:
    ```bash
    ./scripts/segment_all_board_images.py
    ```
    
As example and explanation of how the board segmenter works see this notebook: https://github.com/ivankeller/setgame/blob/master/notebooks/segmentboard_demo.ipynb




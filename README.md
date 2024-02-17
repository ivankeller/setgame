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

### Set the environment
From the root directory of this project do:
```bash
pipenv install
pipenv shell
```

## Run the tests 
```bash
python -m unittest discover -v
```

## 2. Segment cards from board images  
From the root directory of this project do:
```bash
export PYTHONPATH=.
```
Segment cards from a single board image (JPEG or PNG):
```bash
./setgame/segmentboard/scripts/segment_all_board_images.py -i <input board image> -o <output directory>
```
Segment cards for all board images in a given directory:  
- edit file `setgame/segmentboard/scripts/segment_all_board_images.py` adapting `INPUT_DIR` and `OUTPUT_DIR` with the desired paths
- run the script:
    ```bash
    ./setgame/segmentboard/scripts/segment_all_board_images.py
    ```
    
As example and explanation of how the board segmentator works is displayed on this notebook: https://github.com/ivankeller/setgame/blob/master/setgame/notebooks/segmentboard_demo.ipynb

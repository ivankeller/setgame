# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: setgame
#     language: python
#     name: setgame
# ---

# # Annotate Set game cards

# ## config

# +
# examples to annotate: list of path or path to directory with card images
examples = [
    '../../datatest/examples_for_annotator/IMG_20170518_204728_0.png', 
    '../../datatest/examples_for_annotator/IMG_20170518_205001_7.png',
]

#examples = '../../datatest/examples_for_annotator'
# -

# # run the annotation

import sys
sys.path.append('..')

# %load_ext autoreload
# %autoreload 2

from annotator import Annotator
from base_class.label import Label, Color, Number, Shape, Shading

annotator = Annotator(examples)

annotator.examples

annotator.annotate()

annotator.annotations





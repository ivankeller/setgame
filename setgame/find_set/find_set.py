#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# This module implements the game SET which consist in finding sets of 3 cards
# with some special caracteristics among a group of 12.
# See: http://en.wikipedia.org/wiki/Set_(game)
################################################################################

import random


class Card(object):
    """A Card has four features: number, color, shape and shading."""

    numbers  = ['1', '2', '3']
    shadings = ['solid', 'striped', 'open']
    colors   = ['green', 'purple', 'red']
    shapes   = ['diamond', 'squiggle', 'oval']

    def __init__(self, number=None, shading=None, color=None, shape=None):
        if not number:  number  = random.choice(Card.numbers)
        if not shading: shading = random.choice(Card.shadings)
        if not color:   color   = random.choice(Card.colors)
        if not shape:   shape   = random.choice(Card.shapes)
        self.number  = number
        self.shading = shading
        self.color   = color
        self.shape   = shape

    @classmethod
    def from_dict(cls, d: dict) -> 'Card':
        """Create a Card from the dict returned by predict_card().

        Parameters
        ----------
        d : dict
            Output of predict_card(), e.g.:
            {"number": "2", "shape": "oval", "shading": "striped", "color": "red"}

        Returns
        -------
        Card
        """
        return cls(
            number  = d['number'],
            shading = d['shading'],
            color   = d['color'],
            shape   = d['shape'],
        )

    def __repr__(self):
        return repr((self.number, self.shading, self.color, self.shape))

    def __str__(self):
        return ' '.join(self.get_features())

    def get_features(self):
        return [self.number, self.shading, self.color, self.shape]


####### Card manipulation. #####################################################
def gen_all_cards():
    """Generate and return the complete set of 81 cards."""
    return [Card(num, sha, col, sym)
            for num in Card.numbers
            for sha in Card.shadings
            for col in Card.colors
            for sym in Card.shapes]


def remove_sublist(cards, sublist):
    """Remove items of sublist that are in cards (items are unique in both)."""
    for item in sublist:
        assert item in cards, "item not found"
        cards.remove(item)


def sample_cards(cards, nb):
    """Choose nb cards from cards, remove and return them."""
    sample = random.sample(cards, nb)
    remove_sublist(cards, sample)
    return sample


def gen_board(cards, number_of_cards=12):
    """Choose number_of_cards random cards from cards and remove them from cards."""
    board = []
    while number_of_cards > 0:
        random_card = random.choice(cards)
        board.append(random_card)
        cards.remove(random_card)
        number_of_cards -= 1
    return board


####### Set rules ##############################################################
def are_all_equal(items):
    """Check if all items in a non-empty list are equal.

    >>> are_all_equal([1, 1, 1])
    True
    >>> are_all_equal([1, 1, 2])
    False
    >>> are_all_equal([1])
    True
    """
    if len(items) == 0:
        raise ValueError('non-empty list is required as argument.')
    for item in items:
        if item != items[0]:
            return False
    return True


def are_all_different(items):
    """Check if all items in a non-empty list are different.

    >>> are_all_different([1, 2, 3])
    True
    >>> are_all_different([1, 2, 3, 2])
    False
    >>> are_all_different([1])
    False
    """
    if len(items) == 0:
        raise ValueError('non-empty list is required as argument.')
    if len(items) == 1:
        return False
    return len(set(items)) == len(items)


def is_Set(three_cards):
    """Check if three cards form a Set as defined in the Set game rules."""
    for i in range(len(three_cards[0].get_features())):
        feature_values = [card.get_features()[i] for card in three_cards]
        if not (are_all_equal(feature_values) or are_all_different(feature_values)):
            return False
    return True


####### Finding all Sets #######################################################
def _add_to_all_sublists(item, sublists):
    """Add item in front of every sub-list in sublists.

    >>> _add_to_all_sublists(0, [[1], [2], [3]])
    [[0, 1], [0, 2], [0, 3]]
    >>> _add_to_all_sublists('z', [[]])
    [['z']]
    """
    return [[item] + sub for sub in sublists]


def _list_parts(items, p):
    """List all non-ordered p-element subsets of items.

    >>> _list_parts([1, 2, 3], 1)
    [[1], [2], [3]]
    >>> _list_parts([1, 2, 3], 2)
    [[1, 2], [1, 3], [2, 3]]
    >>> _list_parts([1, 2, 3], 0)
    []
    >>> _list_parts([1, 2, 3], 3)
    [[1, 2, 3]]
    >>> _list_parts([], 2)
    []
    """
    if p == 1:
        return [[item] for item in items]
    list_of_parts = []
    for i, it in enumerate(items[:-1]):
        list_of_parts += _add_to_all_sublists(it, _list_parts(items[i+1:], p - 1))
    return list_of_parts


def find_all_Sets(cards):
    """Return all triplets that form a valid Set from a list of cards.

    Returns an empty list if there is no Set.
    """
    return [triplet for triplet in _list_parts(cards, 3) if is_Set(triplet)]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
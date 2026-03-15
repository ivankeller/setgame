import unittest

from setgame.find_set.find_set import (
    Card,
    are_all_different,
    are_all_equal,
    find_all_Sets,
    gen_all_cards,
    is_Set,
    _list_parts,
)


class TestCard(unittest.TestCase):

    def test_explicit_init(self):
        card = Card(number='2', shading='solid', color='red', shape='oval')
        self.assertEqual(card.number,  '2')
        self.assertEqual(card.shading, 'solid')
        self.assertEqual(card.color,   'red')
        self.assertEqual(card.shape,   'oval')

    def test_random_init_valid_values(self):
        card = Card()
        self.assertIn(card.number,  Card.numbers)
        self.assertIn(card.shading, Card.shadings)
        self.assertIn(card.color,   Card.colors)
        self.assertIn(card.shape,   Card.shapes)

    def test_from_dict(self):
        d = {'number': '3', 'shape': 'diamond', 'shading': 'striped', 'color': 'purple'}
        card = Card.from_dict(d)
        self.assertEqual(card.number,  '3')
        self.assertEqual(card.shape,   'diamond')
        self.assertEqual(card.shading, 'striped')
        self.assertEqual(card.color,   'purple')

    def test_get_features_order(self):
        card = Card(number='1', shading='open', color='green', shape='squiggle')
        self.assertEqual(card.get_features(), ['1', 'open', 'green', 'squiggle'])

    def test_str(self):
        card = Card(number='1', shading='open', color='green', shape='squiggle')
        self.assertEqual(str(card), '1 open green squiggle')


class TestAreAllEqual(unittest.TestCase):

    def test_all_equal(self):
        self.assertTrue(are_all_equal(['red', 'red', 'red']))

    def test_not_all_equal(self):
        self.assertFalse(are_all_equal(['red', 'red', 'green']))

    def test_single_element(self):
        self.assertTrue(are_all_equal(['red']))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            are_all_equal([])


class TestAreAllDifferent(unittest.TestCase):

    def test_all_different(self):
        self.assertTrue(are_all_different(['red', 'green', 'purple']))

    def test_not_all_different(self):
        self.assertFalse(are_all_different(['red', 'green', 'red']))

    def test_single_element_is_false(self):
        self.assertFalse(are_all_different(['red']))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            are_all_different([])

    def test_does_not_mutate_input(self):
        items = ['red', 'green', 'purple']
        original = items.copy()
        are_all_different(items)
        self.assertEqual(items, original)


class TestIsSet(unittest.TestCase):

    def _card(self, number, shading, color, shape):
        return Card(number=number, shading=shading, color=color, shape=shape)

    def test_all_same_is_valid(self):
        # All four attributes identical across all three cards
        triplet = [
            self._card('2', 'solid', 'red', 'oval'),
            self._card('2', 'solid', 'red', 'oval'),
            self._card('2', 'solid', 'red', 'oval'),
        ]
        self.assertTrue(is_Set(triplet))

    def test_all_different_is_valid(self):
        # All four attributes all-different across all three cards
        triplet = [
            self._card('1', 'solid',   'red',    'oval'),
            self._card('2', 'striped', 'green',  'diamond'),
            self._card('3', 'open',    'purple', 'squiggle'),
        ]
        self.assertTrue(is_Set(triplet))

    def test_mixed_same_and_different_is_valid(self):
        # number same, others all-different
        triplet = [
            self._card('2', 'solid',   'red',    'oval'),
            self._card('2', 'striped', 'green',  'diamond'),
            self._card('2', 'open',    'purple', 'squiggle'),
        ]
        self.assertTrue(is_Set(triplet))

    def test_two_same_one_different_is_invalid(self):
        # color has two 'red' and one 'green' → invalid
        triplet = [
            self._card('1', 'solid', 'red',   'oval'),
            self._card('2', 'solid', 'red',   'oval'),
            self._card('3', 'solid', 'green', 'oval'),
        ]
        self.assertFalse(is_Set(triplet))


class TestFindAllSets(unittest.TestCase):

    def _card(self, number, shading, color, shape):
        return Card(number=number, shading=shading, color=color, shape=shape)

    def test_finds_known_set(self):
        valid_set = [
            self._card('1', 'solid',   'red',    'oval'),
            self._card('2', 'striped', 'green',  'diamond'),
            self._card('3', 'open',    'purple', 'squiggle'),
        ]
        # Add noise cards that form no set with each other
        noise = [
            self._card('1', 'solid', 'red',   'diamond'),
            self._card('1', 'solid', 'green', 'diamond'),
        ]
        results = find_all_Sets(valid_set + noise)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any(set(map(id, r)) == set(map(id, valid_set)) for r in results))

    def test_returns_empty_when_no_set(self):
        # Three identical cards: all-same for every attribute → actually a valid set.
        # Build an explicit non-set triplet instead:
        # two 'red' and one 'green' on color → invalid
        cards = [
            self._card('1', 'solid', 'red',   'oval'),
            self._card('2', 'solid', 'red',   'oval'),
            self._card('3', 'solid', 'green', 'oval'),
        ]
        self.assertEqual(find_all_Sets(cards), [])

    def test_full_deck_has_sets(self):
        # A full deck of 81 cards always contains many sets
        all_cards = gen_all_cards()
        results = find_all_Sets(all_cards)
        self.assertGreater(len(results), 0)


class TestListParts(unittest.TestCase):

    def test_triplets(self):
        parts = _list_parts([1, 2, 3, 4], 3)
        self.assertEqual(len(parts), 4)   # C(4,3) = 4
        self.assertIn([1, 2, 3], parts)
        self.assertIn([1, 2, 4], parts)

    def test_pairs(self):
        parts = _list_parts([1, 2, 3], 2)
        self.assertEqual(len(parts), 3)   # C(3,2) = 3

    def test_singles(self):
        parts = _list_parts([1, 2, 3], 1)
        self.assertEqual(parts, [[1], [2], [3]])

    def test_empty_input(self):
        self.assertEqual(_list_parts([], 2), [])


if __name__ == '__main__':
    unittest.main()
import unittest
from src import *


class Test(unittest.TestCase):
    def test_integer_add(self):
        self.assertEqual(add_two_numbers(2, 3), 5)
    def test_integer_plus_float(self):
        self.assertEqual(add_two_numbers(1.1, 2), 3.1)

if __name__ == '__main__':
    unittest.main()

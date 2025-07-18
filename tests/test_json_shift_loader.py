import unittest
import numpy as np
from json_shift_loader import load_shift_patterns

class LoaderTest(unittest.TestCase):
    def test_v1_format(self):
        data = load_shift_patterns('examples/shift_config.json')
        self.assertTrue(data)
        for arr in data.values():
            self.assertEqual(arr.shape, (7*24,))

    def test_v2_format(self):
        data = load_shift_patterns('examples/shift_config_v2.json')
        self.assertTrue(data)
        for arr in data.values():
            self.assertEqual(arr.shape, (7*24,))

if __name__ == '__main__':
    unittest.main()

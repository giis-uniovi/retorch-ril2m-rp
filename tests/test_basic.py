# -*- coding: utf-8 -*-
import logging
import unittest

import ril2m

logger = logging.getLogger(__name__)


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_triangle_isosceles(self):
        logger.debug("Starting the test: " + self._testMethodName)
        assert ril2m.get_type_triangle(5, 5, 2) == "ISOSCELES"
        logger.debug("Ending the test: " + self._testMethodName)

    def test_triangle_equilateral(self):
        logger.debug("Starting the test: " + self._testMethodName)
        assert ril2m.get_type_triangle(5, 5, 5) == "EQUILATERAL"
        logger.debug("Ending the test: " + self._testMethodName)

    def test_triangle_scalene(self):
        logger.debug("Starting the test: " + self._testMethodName)
        assert ril2m.get_type_triangle(3, 5, 2) == "SCALENE"
        logger.debug("Ending the test: " + self._testMethodName)


if __name__ == '__main__':
    unittest.main()

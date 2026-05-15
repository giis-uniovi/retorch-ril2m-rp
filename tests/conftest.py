# conftest.py
import logging

logger = logging.getLogger(__name__)


def pytest_runtest_setup(item):
    logger.debug("Starting setup of the Test: " + item.name)

    logger.debug("Ending setup of the Test: " + item.name)


def pytest_runtest_teardown(item):
    logger.debug("Starting teardown of the Test: " + item.name)

    logger.debug("Ending teardown of the Test: " + item.name)

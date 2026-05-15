# -*- coding: utf-8 -*-
"""File I/O utilities."""

import logging
import os

logger = logging.getLogger(__name__)

_OUTPUT_DIR = os.path.normpath(os.path.join(os.getcwd(), "outputs", "python"))


def loadfile(route: str) -> str:
    """Load a file into a string and return it."""
    try:
        with open(route, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {route}"
    except OSError as e:
        return f"Reading failure: {e}"


def save_output_to_file(filename: str, content: str, output_dir: str = _OUTPUT_DIR) -> None:
    """Save LLM output to a file under *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("File saved at: %s", output_path)

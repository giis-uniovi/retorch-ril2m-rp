# -*- coding: utf-8 -*-
"""Code snippet extraction utilities."""

import re


def extract_snippets(message: str, language: str | None = None) -> str:
    """Extract fenced code blocks from *message*.

    If *language* is given, only blocks tagged with that language are returned.
    """
    if language:
        pattern = rf"```{language}\n(.*?)```"
    else:
        pattern = r"```(?:\w+\n)?(.*?)```"
    blocks = re.findall(pattern, message, re.DOTALL)
    return "\n".join(b.strip() for b in blocks)

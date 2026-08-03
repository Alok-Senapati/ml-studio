"""
Base exception hierarchy for ml-core.
"""

from __future__ import annotations


class MLCoreError(Exception):
    """
    Base exception class for all errors raised by the ml-core library.

    All custom exceptions in ml-core inherit from this class to allow users
    to catch library-specific errors cleanly.

    Notes
    -----
    Standard Python `Exception` acts as the direct base class for `MLCoreError`.
    """

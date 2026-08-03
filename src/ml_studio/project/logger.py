"""
Logging configuration for ml-studio CLI tools.
"""

from __future__ import annotations

import logging

# Configure basic logging format displaying raw message string
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

#: Shared Logger instance for ml-studio project generation operations
logger = logging.getLogger("ml_studio")

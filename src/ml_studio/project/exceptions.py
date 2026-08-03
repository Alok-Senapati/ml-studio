"""
Custom exception hierarchy for project generator operations.
"""

from __future__ import annotations


class ProjectGeneratorError(Exception):
    """
    Base exception for all errors raised during project generation.

    Notes
    -----
    Acts as the parent exception for invalid project names, existing destination
    paths, and missing template directories.
    """


class InvalidProjectNameError(ProjectGeneratorError):
    """
    Raised when the requested project name violates naming constraints.

    Raised if the name is empty, not a valid Python identifier, a reserved keyword,
    or does not conform to `snake_case`.
    """


class ProjectAlreadyExistsError(ProjectGeneratorError):
    """
    Raised when the target project destination directory already exists.
    """


class TemplateNotFoundError(ProjectGeneratorError):
    """
    Raised when the required project template directory cannot be found.
    """

from __future__ import annotations


class ProjectGeneratorError(Exception):
    """Base exception for project generation."""


class InvalidProjectNameError(ProjectGeneratorError):
    """Raised when the project name is invalid."""


class ProjectAlreadyExistsError(ProjectGeneratorError):
    """Raised when the destination project already exists."""


class TemplateNotFoundError(ProjectGeneratorError):
    """Raised when the template directory cannot be found."""

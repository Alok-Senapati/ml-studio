"""
Configuration settings container for __PROJECT_NAME__.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Project settings schema container.
    """


#: Application settings instance
settings = Settings()

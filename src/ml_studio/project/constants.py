"""
Global path constants and placeholder mapping dictionaries for project generation.
"""

from __future__ import annotations

from pathlib import Path

#: Root directory path of the ml-studio repository
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Directory containing the default project template scaffolding
TEMPLATE_DIR = PROJECT_ROOT / "templates" / "ml_template"

#: Target output directory where newly scaffolded projects are generated
PROJECTS_DIR = PROJECT_ROOT / "projects"

#: Standard placeholder key mappings replaced during project generation
PLACEHOLDERS = {
    "__PROJECT_NAME__": "project_name",
    "__PROJECT_DESCRIPTION__": "project_description",
    "__AUTHOR__": "author",
    "__YEAR__": "year",
}
